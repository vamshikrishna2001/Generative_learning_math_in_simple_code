import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    def __init__(self, input_size, output_size, num_control_points=3):
        super(KANLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_control_points = num_control_points
        
        # Initialize control points (learnable) - cover expected input range
        # For inputs like [12, 32, 43, 54], set range to cover [10, 60]
        control_points_init = torch.linspace(10, 60, num_control_points).unsqueeze(0).unsqueeze(0).repeat(
            input_size, output_size, 1
        )
        self.control_points = nn.Parameter(control_points_init)
        
        # Initialize spline coefficients (learnable) - small random values
        self.spline_coeffs = nn.Parameter(
            torch.randn(input_size, output_size, num_control_points) * 0.1
        )
        
        # Initialize weights (learnable) - Xavier initialization
        self.weights = nn.Parameter(
            torch.randn(input_size, output_size) * math.sqrt(2.0 / input_size)
        )
    
    def b_spline_basis(self, x, control_points):
        """
        Simple B-spline basis functions using Gaussian-like shapes
        x: [batch_size, input_size]
        control_points: [input_size, output_size, num_control_points]
        Returns: [batch_size, input_size, output_size, num_control_points]
        """
        batch_size = x.shape[0]
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(2).unsqueeze(3)  # [batch, input, 1, 1]
        cp_expanded = control_points.unsqueeze(0)  # [1, input, output, control_points]
        
        # Gaussian basis functions centered at control points
        sigma = 0.5  # Width of basis functions (matching manual calculation)
        basis = torch.exp(-((x_expanded - cp_expanded) ** 2) / (2 * sigma ** 2))
        
        # Normalize so each basis function sums to reasonable values
        basis = basis / (sigma * math.sqrt(2 * math.pi))
        
        return basis
    
    def forward(self, x):
        """
        x: [batch_size, input_size]
        Returns: [batch_size, output_size]
        """
        batch_size = x.shape[0]
        
        # SiLU activation
        silu_x = F.silu(x)  # [batch_size, input_size]
        
        # Compute basis functions
        basis = self.b_spline_basis(x, self.control_points)  # [batch, input, output, control_points]
        
        # Compute spline values: sum(c_i * b_i(x))
        spline_coeffs_expanded = self.spline_coeffs.unsqueeze(0)  # [1, input, output, control_points]
        spline_values = torch.sum(spline_coeffs_expanded * basis, dim=-1)  # [batch, input, output]
        
        # KAN edge function: φ(x) = w * (silu(x) + spline(x))
        silu_expanded = silu_x.unsqueeze(2)  # [batch, input, 1]
        weights_expanded = self.weights.unsqueeze(0)  # [1, input, output]
        
        phi_values = weights_expanded * (silu_expanded + spline_values)  # [batch, input, output]
        
        # Sum over input dimensions for each output node
        output = torch.sum(phi_values, dim=1)  # [batch, output]
        
        return output

# Example usage
if __name__ == "__main__":
    # Example data from the step-by-step calculation
    x = torch.tensor([[12.0, 32.0, 43.0, 54.0]])  # [batch=1, input=4]
    print(f"Input: {x}")
    print(f"Input shape: {x.shape}")
    print()
    
    # Create KAN layer: 4 inputs -> 2 outputs (matching the example)
    kan_layer = KANLayer(input_size=4, output_size=2, num_control_points=3)
    
    print(f"Control points shape: {kan_layer.control_points.shape}")
    print(f"Spline coefficients shape: {kan_layer.spline_coeffs.shape}")
    print(f"Weights shape: {kan_layer.weights.shape}")
    print()
    
    # Show initialized parameters
    print("INITIALIZED PARAMETERS:")
    print(f"Control points for edge (0,0): {kan_layer.control_points[0, 0]}")
    print(f"Spline coefficients for edge (0,0): {kan_layer.spline_coeffs[0, 0]}")
    print(f"Weight for edge (0,0): {kan_layer.weights[0, 0].item():.3f}")
    print()
    
    # Forward pass demonstration
    print("="*50)
    print("KAN FORWARD PASS")
    print("="*50)
    
    output = kan_layer(x)
    print(f"Input:  {x.squeeze().tolist()}")
    print(f"Output: {output.squeeze().tolist()}")
    print()
    
    # Training demonstration
    print("="*50)
    print("TRAINING EXAMPLE")
    print("="*50)
    
    # Create target
    target = torch.tensor([[100.0, 150.0]])  # [batch=1, output=2]
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(kan_layer.parameters(), lr=0.01)
    
    print(f"Target: {target.squeeze().tolist()}")
    print()
    
    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        
        output = kan_layer(x)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:2d}: Loss = {loss.item():8.3f}, Output = {output.squeeze().tolist()}")
    
    print()
    print("LEARNED PARAMETERS:")
    print(f"Control points for edge (0,0): {kan_layer.control_points[0, 0].detach()}")
    print(f"Spline coefficients for edge (0,0): {kan_layer.spline_coeffs[0, 0].detach()}")
    print(f"Weight for edge (0,0): {kan_layer.weights[0, 0].item():.3f}")

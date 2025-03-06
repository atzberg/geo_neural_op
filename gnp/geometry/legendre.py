import torch
from scipy.special import legendre
from typing import Optional

class Legendre1D:
    """
    Class to compute the Legendre polynomials and their derivatives
    """    
    def __init__(self, degree: int=3):
        self.degree = degree
        self.legendres = [torch.Tensor(legendre(i).coef) for i in range(degree + 1)]
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Evaluate the Legendre polynomials at the input points.

        Parameters
        ----------
        x : torch.Tensor
            Input points.

        Returns
        -------
        torch.Tensor
            Legendre polynomials evaluated at the input points.
        """        
        xs = torch.stack([x.pow(i) for i in range(self.degree, -1, -1)], dim=-1)
        xs = torch.stack(
            [(self.legendres[i].to(x.device) * xs[..., -i-1:]).sum(dim=-1) 
             for i in range(self.degree+1)], dim=-1)
        return xs

    def first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Compute the derivative of the Legendre polynomials at the input points.

        Parameters
        ----------
        x : torch.Tensor
            Input points.

        Returns
        -------
        torch.Tensor
            Derivative of Legendre polynomials evaluated at the input points.
        """ 
        xs = torch.stack([i * x.pow(i-1) for i in range(self.degree, -1, -1)], dim=-1)
        xs = torch.nan_to_num(xs, nan=0.0)
        dxs = torch.stack(
            [(self.legendres[i].to(x.device) * xs[..., -i-1:]).sum(dim=-1) 
             for i in range(self.degree+1)], dim=-1)
        return dxs

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the second derivative of the Legendre polynomials at the input points

        Parameters
        ----------
        x : torch.Tensor
            Input points.

        Returns
        -------
        torch.Tensor
            Second derivative of Legendre polynomials evaluated at the input points.
        """        
        xs = torch.stack([i * (i-1) * x.pow(i-2) for i in range(self.degree, -1, -1)], dim=-1)
        xs = torch.nan_to_num(xs, nan=0.0)
        ddxs = torch.stack(
            [(self.legendres[i].to(x.device) * xs[..., -i-1:]).sum(dim=-1) 
             for i in range(self.degree+1)], dim=-1)
        return ddxs
    
class Legendre2D:
    def __init__(self, degree: int=3):
        self.degree = degree
        self.legendre1d = Legendre1D(degree=self.degree)

    def evaluate(self, xy_data: torch.Tensor) -> torch.Tensor:
        
        """
        Evaluate tensor product of Legendre polynomials at the input points.

        Parameters
        ----------
        xy_data : torch.Tensor
            (n, 2) tensor of input points.

        Returns
        -------
        torch.Tensor
            (n, (degree + 1) ** 2) tensor of Legendre polynomials evaluated at the input points.
        """
        x = xy_data[..., 0]
        y = xy_data[..., 1]
        size = x.size()
        xs = self.legendre1d.evaluate(x).view(*size, 1, self.degree + 1)
        ys = self.legendre1d.evaluate(y).view(*size, self.degree + 1, 1)
        legendre_values = xs * ys
        return legendre_values.flatten(start_dim=-2)
    
    def evaluate_from_coeffs(self, 
                             xy_data: torch.Tensor, 
                             coeffs: torch.Tensor, 
                             batch: Optional[torch.LongTensor]=None) -> torch.Tensor:
        """
        Evaluate a function at given points from its Legendre coefficients

        Args:
            xy_data (torch.Tensor): (n, 2) tensor of input points
            coeffs (torch.Tensor): tensor of legendre coefficients
            batch (Optional[torch.LongTensor], optional): batch indices for parallel
                computation over many batches. Defaults to None.

        Returns:
            torch.Tensor: (n, 1) tensor of function values at the input points
        """        
        legendre_values = self.evaluate(xy_data)
        return self.compute_from_coeffs(legendre_values, coeffs, batch)

    def compute_from_coeffs(self, 
                            values: torch.Tensor, 
                            coeffs: torch.Tensor, 
                            batch: Optional[torch.LongTensor]=None) -> torch.Tensor:
        """
        Compute function values using Legendre coefficients.

        Parameters
        ----------
        values : torch.Tensor
            Values of the Legendre polynomials (or derivatives) at the input points.
        coeffs : torch.Tensor
            Legendre basis coefficients.
        batch : Optional[torch.LongTensor], optional
            Batch indices. Defaults to None.

        Returns
        -------
        torch.Tensor
            (n, 1) tensor of function values at the input points.
        """    
            
        if batch is None:
            return (coeffs * values).sum(dim=-1).unsqueeze(-1)
        else:
            return (coeffs[batch] * values).sum(dim=-1).unsqueeze(-1)

    def evaluate_derivatives(self, xy_data: torch.TensorType) -> tuple[torch.Tensor]:
        """
        Evaluate the derivatives of the Legendre polynomials at the input points.

        Parameters
        ----------
        xy_data : torch.Tensor
            (n, 2) tensor of input points.

        Returns
        -------
        tuple of torch.Tensor
            Derivatives of the Legendre polynomials evaluated at the input points.
        """  
              
        x = xy_data[..., 0]
        y = xy_data[..., 1]
        size = x.size()

        xs = self.legendre1d.evaluate(x).view(*size, 1, self.degree + 1)
        ys = self.legendre1d.evaluate(y).view(*size, self.degree + 1, 1)

        dxs = self.legendre1d.first_derivative(x).view(*size, 1, self.degree + 1)
        dys = self.legendre1d.first_derivative(y).view(*size, self.degree + 1, 1)

        ddxs = self.legendre1d.second_derivative(x).view(*size, 1, self.degree + 1)
        ddys = self.legendre1d.second_derivative(y).view(*size, self.degree + 1, 1)

        dx = (dxs * ys).squeeze(0).flatten(start_dim=-2)
        dy = (xs * dys).squeeze(0).flatten(start_dim=-2)

        dxdy = (dxs * dys).squeeze(0).flatten(start_dim=-2)
        dxdx = (ddxs * ys).squeeze(0).flatten(start_dim=-2)
        dydy = (xs * ddys).squeeze(0).flatten(start_dim=-2)

        return dx, dy, dxdy, dxdx, dydy

    def derivatives_from_coeffs(self, 
                                xy_data: torch.Tensor, 
                                coeffs: torch.Tensor, 
                                batch: Optional[torch.LongTensor]=None) -> torch.Tensor:
        """
        Compute the derivatives of a function at given points from its Legendre coefficients.

        Parameters
        ----------
        xy_data : torch.Tensor
            (n, 2) tensor of input points.
        coeffs : torch.Tensor
            Tensor of Legendre coefficients.
        batch : Optional[torch.LongTensor], optional
            Batch indices for parallel computation over many batches. Defaults to None.

        Returns
        -------
        torch.Tensor
            (n, 5) tensor of function derivatives at the input points.
        """
        
        dx, dy, dxdy, dxdx, dydy = self.evaluate_derivatives(xy_data)
        derivatives = torch.cat((self.compute_from_coeffs(dx, coeffs, batch),
                                 self.compute_from_coeffs(dy, coeffs, batch),
                                 self.compute_from_coeffs(dxdy, coeffs, batch),
                                 self.compute_from_coeffs(dxdx, coeffs, batch),
                                 self.compute_from_coeffs(dydy, coeffs, batch)),
                                dim=-1)

        return derivatives

    def batch_ls_best_fit(self, 
                          x: torch.Tensor, 
                          batch: Optional[torch.LongTensor]=None) -> torch.Tensor:
        """
        Compute the least squares best fit of the Legendre coefficients.

        Parameters
        ----------
        x : torch.Tensor
            (n, 2) tensor of input points.
        batch : Optional[torch.LongTensor], optional
            Batch indices. Defaults to None.

        Returns
        -------
        torch.Tensor
            (num_batches, (degree + 1) ** 2) tensor of Legendre coefficients.
        """        
        
        device = x.device
        x = x.cpu()
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        batch = batch.cpu()
        num_batches = batch.max().item() + 1
        legvals = [self.evaluate(x[batch == i]) for i in range(num_batches)]
        xs = [x[batch == i, 2].view(-1, 1) for i in range(num_batches)]
        out = torch.cat(
            [torch.linalg.lstsq(A, b, driver='gelsd').solution.view(1, -1) for A, b in zip(legvals, xs)], dim=0
            ).to(device)
        return out
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt 
import pathlib 

class DispersionData:
    def __init__(self, file_path: str = None):
        """
        Initialize the DispersionData object.
        
        Args:
            file_path (str, optional): Path to CSV file containing dispersion data. 
                                      If None, creates empty object.
        """
        self.modes = {}  # Dictionary to store data for each mode
        self.all_modes = set()  # Set of all available modes
        
        if file_path:
            self.load_from_csv(file_path)
    
    def load_from_csv(self, file_path: str):
        """
        Load dispersion data from a CSV file.
        
        Args:
            file_path (str): Path to CSV file containing dispersion data.
        """
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract all unique mode identifiers from column headers
        mode_identifiers = set()
        for col in df.columns:
            mode = col.split()[0]  # Extract mode identifier (e.g., 'A0', 'A1')
            mode_identifiers.add(mode)
        
        self.all_modes.update(mode_identifiers)
        
        # Process each mode
        for mode in mode_identifiers:
            # Select columns for this mode
            mode_cols = [col for col in df.columns if col.startswith(mode)]
            
            # Create a DataFrame for this mode
            mode_df = df[mode_cols].copy()
            
            # Rename columns by removing the mode prefix
            mode_df.columns = [col[len(mode)+1:] for col in mode_df.columns]
            
            # Store the data
            self.modes[mode] = mode_df
    
    def get_mode_data(self, mode: str) -> Optional[pd.DataFrame]:
        """
        Get dispersion data for a specific mode.
        
        Args:
            mode (str): Mode identifier (e.g., 'A0', 'S1')
            
        Returns:
            pd.DataFrame: Data for the requested mode, or None if not found
        """
        return self.modes.get(mode, None)
    
    def get_available_modes(self) -> List[str]:
        """
        Get list of all available modes in the dataset.
        
        Returns:
            List[str]: List of mode identifiers
        """
        return list(self.modes.keys())
    
    def get_dispersion_curves(self, frequency_range: Tuple[float, float] = None, 
                             modes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get dispersion curves for specified modes within a frequency range.
        
        Args:
            frequency_range (Tuple[float, float], optional): (min_freq, max_freq) in kHz.
                If None, returns all frequencies.
            modes (List[str], optional): List of mode identifiers to include.
                If None, includes all available modes.
                
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping mode names to their dispersion data
        """
        if modes is None:
            modes = self.get_available_modes()
        
        result = {}
        
        for mode in modes:
            mode_data = self.get_mode_data(mode)
            if mode_data is None:
                continue
                
            if frequency_range is None:
                result[mode] = mode_data
            else:
                min_freq, max_freq = frequency_range
                # Find the frequency column (it's usually 'f (kHz)')
                freq_col = [col for col in mode_data.columns if 'f' in col.lower()][0]
                mask = (mode_data[freq_col] >= min_freq) & (mode_data[freq_col] <= max_freq)
                result[mode] = mode_data[mask].copy()
        
        return result
    
    def add_mode_data(self, mode: str, data: pd.DataFrame):
        """
        Add or update data for a specific mode.
        
        Args:
            mode (str): Mode identifier
            data (pd.DataFrame): Dispersion data for the mode
        """
        self.modes[mode] = data.copy()
        self.all_modes.add(mode)
    
    def merge(self, other: 'DispersionData'):
        """
        Merge data from another DispersionData object into this one.
        
        Args:
            other (DispersionData): Another DispersionData instance
        """
        for mode in other.get_available_modes():
            self.add_mode_data(mode, other.get_mode_data(mode))

    def plot(self, x_header: str, y_header: str, modes: Union[str, List[str]] = None, frequency_range: Tuple[float, float] = None, 
             ax: plt.Axes = None, **plot_kwargs) -> plt.Axes:
        """
        Plot specified data headers for selected modes.
        
        Args:
            x_header (str): Header for x-axis data (e.g., 'f (kHz)')
            y_header (str): Header for y-axis data (e.g., 'Phase velocity (m/ms)')
            modes (Union[str, List[str]], optional): Mode(s) to plot. If None, plots all available modes.
            frequency_range (Tuple[float, float], optional): Frequency range to plot (min_freq, max_freq).
            ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure.
            **plot_kwargs: Additional keyword arguments passed to matplotlib's plot function.
            
        Returns:
            plt.Axes: The matplotlib axes object containing the plot
        """
        # Handle single mode vs list of modes
        if modes is None:
            modes = self.get_available_modes()
        elif isinstance(modes, str):
            modes = [modes]
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the data for each mode
        curves = self.get_dispersion_curves(frequency_range=frequency_range, modes=modes)
        
        # Plot each mode
        for mode, data in curves.items():
            # Find the exact column names (case-insensitive and allowing for slight variations)
            x_col = self._find_matching_column(data.columns, x_header)
            y_col = self._find_matching_column(data.columns, y_header)
            
            if x_col is None or y_col is None:
                print(f"Warning: Could not find both headers in mode {mode}. Skipping.")
                continue
            
            # Plot with mode name in label
            ax.plot(data[x_col], data[y_col], label=mode, **plot_kwargs)
        
        # Add labels and legend
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.grid(True)
        ax.legend()
        
        return ax
    
    def _find_matching_column(self, columns: List[str], header: str) -> Optional[str]:
        """
        Helper method to find a matching column name with flexible matching.
        
        Args:
            columns (List[str]): List of available column names
            header (str): Header to match against
            
        Returns:
            Optional[str]: The matching column name, or None if not found
        """
        header_lower = header.lower()
        for col in columns:
            if header_lower in col.lower():
                return col
        return None
    
    def get_value(self, mode: Union[str, Tuple[str]], frequency: float, target_header: str, 
             interpolation: str = 'linear') -> Union[float, Tuple[float]]:
        """
        Get a value from a specified column at a given frequency for one or more modes.
        
        Args:
            mode (Union[str, Tuple[str]]): Mode identifier or tuple of identifiers (e.g., 'A0', ('A0', 'S1'))
            frequency (float): Frequency in kHz to query
            target_header (str): Header of the column to get value from
            interpolation (str): Interpolation method ('linear', 'nearest', 'spline')
                            Default is 'linear'
                            
        Returns:
            Union[float, Tuple[float]]: The interpolated value(s) at the specified frequency
            
        Raises:
            ValueError: If mode(s) or headers aren't found
        """
        # Handle single mode vs tuple of modes
        if isinstance(mode, str):
            modes = [mode]
            return_tuple = False
        else:
            modes = mode
            return_tuple = True
        
        results = []
        for current_mode in modes:
            # Get mode data
            mode_data = self.get_mode_data(current_mode)
            if mode_data is None:
                raise ValueError(f"Mode '{current_mode}' not found in dataset")
            
            # Find frequency column
            freq_col = self._find_matching_column(mode_data.columns, 'f (kHz)')
            if freq_col is None:
                raise ValueError(f"Frequency column not found in mode '{current_mode}' data")
            
            # Find target column
            target_col = self._find_matching_column(mode_data.columns, target_header)
            if target_col is None:
                raise ValueError(f"Target header '{target_header}' not found in mode '{current_mode}' data")
            
            # Extract data
            freq_data = mode_data[freq_col].values
            target_data = mode_data[target_col].values
            
            # Remove NaN values
            mask = ~np.isnan(freq_data) & ~np.isnan(target_data)
            freq_data = freq_data[mask]
            target_data = target_data[mask]
            
            if len(freq_data) == 0:
                raise ValueError(f"No valid data points available for mode '{current_mode}'")
            
            # Handle interpolation
            if interpolation == 'nearest':
                idx = np.argmin(np.abs(freq_data - frequency))
                results.append(target_data[idx])
            elif interpolation == 'linear':
                results.append(np.interp(frequency, freq_data, target_data))
            elif interpolation == 'spline':
                from scipy import interpolate
                spline = interpolate.CubicSpline(freq_data, target_data, extrapolate=False)
                results.append(spline(frequency))
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation}")
        
        return tuple(results) if return_tuple else results[0]


if __name__=='__main__':
    data_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 's355j2_dispersion_curves'
    
    dispersion = DispersionData()
    for curves_file in data_dir.glob('*.txt'):
        print(curves_file)
        dispersion.merge(DispersionData(curves_file))


    # Get available modes
    print("Available modes:", dispersion.get_available_modes())

    # 3. Get data for a specific mode
    a0_data = dispersion.get_mode_data('A0')

    print("A0 data:", a0_data.head())

    # 4. Get dispersion curves for a frequency range
    frequency_range = (100, 500)  # kHz
    curves = dispersion.get_dispersion_curves(frequency_range=frequency_range, modes=['A0', 'A1'])
    

    # 5. Access specific curves
    a0_phase_velocity = curves['A0']['Phase velocity (m/ms)']
    a0_frequencies = curves['A0']['f (kHz)']

    ax = dispersion.plot('f (kHz)', 'phase velocity', modes=['A0', 'A1', 'S0', 'S1', 'SSH0']) 
    plt.show()
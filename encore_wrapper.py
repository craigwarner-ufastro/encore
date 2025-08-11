import os
import subprocess
import tempfile
import numpy as np
from typing import Union, List, Dict, Any, Tuple
from threading import Thread
import glob

# We'll use astropy to handle FITS files.
try:
    from astropy.io import fits
except ImportError:
    print("Warning: astropy is not installed. FITS file support will be disabled.")
    fits = None

class EncoreWrapper:
    """
    A Python wrapper for the C++ and CUDA `encore` codebase.

    This class handles various input data types, converts them to the required
    4-column ASCII format, executes the `encore` program, and cleans up
    temporary files.
    """

    def __init__(self, encore_executable_path: str = "encore"):
        """
        Initializes the wrapper.

        Args:
            encore_executable_path (str): The path to the `encore` executable.
                                          Defaults to "encore", assuming it's
                                          in the system's PATH.
        """
        self.encore_executable_path = encore_executable_path

    def _convert_to_ascii(self, data: np.ndarray, temp_file_path: str):
        """
        Internal method to save a 4-column numpy array to a temporary
        ASCII file.

        Args:
            data (np.ndarray): A NumPy array with at least 4 columns.
            temp_file_path (str): The path to the temporary file to write.
        """
        if data.shape[1] < 4:
            raise ValueError("Input data must have at least 4 columns.")
        
        # Save the first 4 columns to the temporary file.
        np.savetxt(temp_file_path, data[:, :4], fmt='%1.6f')

    def _stream_output(self, pipe, callback):
        """
        Internal method to stream output from a subprocess pipe line by line.
        """
        for line in iter(pipe.readline, ''):
            callback(line)
        pipe.close()

    def run_correlation(
        self,
        input_data: Union[str, np.ndarray, fits.fitsrec.FITS_rec] = None,
        random_np: int = None,
        rmin: float = 0.0,
        rmax: float = 200.0,
        nside: int = 50,
        boxsize: float = 400.0,
        rescale: float = 1.0,
        outstring: str = 'sample', # Default value for outstring
        savename: str = None,
        loadname: str = None,
        balance_weights: bool = False,
        invert_weights: bool = False,
        column_names: List[str] = None,
        gpu_mode: int = 0,
        gpu_float: bool = False,
        gpu_mixed: bool = False,
        gpu_global: bool = False,
        gpu_mpkernel: int = None,
        only_2pcf: bool = False,
        encore_args: List[str] = [],
        temp_dir: str = None
    ) -> Tuple[bool, int, List[str]]:
        """
        Runs the `encore` correlation function with the provided data.

        This method supports multiple input types:
        - A string representing the path to an existing 4-column ASCII file.
        - A NumPy array with at least 4 columns.
        - An Astropy FITS table object.
        - A string representing the path to a FITS file.
        - You can also generate random particles by using the `random_np` parameter.

        The method handles the necessary data conversion and temporary file
        management. The output from `encore` is now streamed directly to the
        console as it runs.

        Args:
            input_data (Union[str, np.ndarray, FITS_rec]): The input data.
                                                            Required unless `random_np` is used.
            random_np (int): If provided, ignores `input_data` and generates
                             this many random periodic points instead.
            rmin (float): The minimum radius of the smallest pair bin.
            rmax (float): The maximum radius of the largest pair bin.
            nside (int): The grid size for accelerating the pair count.
            boxsize (float): The periodic size of the computational domain.
            rescale (float): How much to dilate the input positions by.
            outstring (str): String to prepend to the output file.
            savename (str): Filename to store the multipoles to.
            loadname (str): Filename to load multipoles from.
            balance_weights (bool): Rescale negative weights so total weight is zero.
            invert_weights (bool): Multiply all weights by -1.
            column_names (List[str]): List of 4 column names to use from a FITS
                                      table (e.g., ['RA', 'DEC', 'Z', 'WEIGHT']).
            gpu_mode (int): GPU mode (0=CPU, 1=GPU, 2+=alt. kernel). Requires
                            compilation in GPU mode.
            gpu_float (bool): Use floats to speed up GPU mode.
            gpu_mixed (bool): Use mixed precision in GPU mode.
            gpu_global (bool): Use global memory always in GPU mode.
            gpu_mpkernel (int): Kernel for multipoles and pairs.
            only_2pcf (bool): Only calculate 2PCF and exit.
            encore_args (List[str]): A list of additional, non-wrapped
                                     command-line arguments for the `encore` program.
            temp_dir (str, optional): A directory for temporary files.
                                      Defaults to the system's temp directory.

        Returns:
            Tuple[bool, int, List[str]]: A tuple containing a boolean success flag,
                                         the return code, and a list of output filenames.
        """
        temp_file = None
        input_file_path = None
        output_files = []

        try:
            # 1. Handle different input data types
            if random_np is not None:
                # Use the random particle generator, ignoring input_data
                command = [self.encore_executable_path, '-ran', str(random_np)]
                input_file_path = None # No input file needed for this mode
            
            else:
                if input_data is None:
                    raise ValueError("Either `input_data` or `random_np` must be provided.")
                
                if isinstance(input_data, str):
                    if input_data.lower().endswith(('.fits', '.fit', '.fits.gz')):
                        if fits is None:
                            raise ImportError("astropy is required for FITS file support.")

                        with fits.open(input_data) as hdul:
                            data_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                            
                            if column_names and len(column_names) == 4:
                                data_columns = [data_hdu.data[col] for col in column_names]
                                data_np = np.stack(data_columns, axis=1)
                            else:
                                data = data_hdu.data
                                data_np = np.array(data.tolist())
                        
                        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix=".ascii")
                        self._convert_to_ascii(data_np, temp_file.name)
                        input_file_path = temp_file.name
                    else:
                        input_file_path = input_data

                elif isinstance(input_data, np.ndarray):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix=".ascii")
                    self._convert_to_ascii(input_data, temp_file.name)
                    input_file_path = temp_file.name

                elif fits is not None and isinstance(input_data, fits.fitsrec.FITS_rec):
                    if column_names and len(column_names) == 4:
                        data_columns = [input_data[col] for col in column_names]
                        data_np = np.stack(data_columns, axis=1)
                    else:
                        data_np = np.array(input_data.tolist())
                    
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix=".ascii")
                    self._convert_to_ascii(data_np, temp_file.name)
                    input_file_path = temp_file.name

                else:
                    raise TypeError(
                        f"Unsupported input type: {type(input_data)}. "
                        "Expected str (file path), np.ndarray, FITS_rec, or a valid `random_np`."
                    )
                
                command = [self.encore_executable_path, '-in', input_file_path]


            # 2. Build the command from the arguments
            command.extend(['-rmin', str(rmin)])
            command.extend(['-rmax', str(rmax)])
            command.extend(['-nside', str(nside)])
            command.extend(['-boxsize', str(boxsize)])
            command.extend(['-rescale', str(rescale)])

            # The outstring parameter now has a default value 'sample'
            command.extend(['-outstr', outstring])
            if savename:
                command.extend(['-save', savename])
            if loadname:
                command.extend(['-load', loadname])
            if balance_weights:
                command.append('-balance')
            if invert_weights:
                command.append('-invert')
            
            # GPU-specific arguments
            if gpu_mode > 0:
                command.extend(['-gpu', str(gpu_mode)])
            if gpu_float:
                command.append('-float')
            if gpu_mixed:
                command.append('-mixed')
            if gpu_global:
                command.append('-global')
            if gpu_mpkernel is not None:
                command.extend(['-mpkernel', str(gpu_mpkernel)])
            if only_2pcf:
                command.append('-2pcf')
            
            # Add any extra arguments provided by the user
            command.extend(encore_args)

            try:
                # 3. Use Popen to run the command and stream output
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                    stdout_lines = []
                    stderr_lines = []

                    stdout_thread = Thread(target=self._stream_output, args=(process.stdout, lambda line: (print(line, end=''), stdout_lines.append(line))))
                    stderr_thread = Thread(target=self._stream_output, args=(process.stderr, lambda line: (print(line, end=''), stderr_lines.append(line))))
                    
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    stdout_thread.join()
                    stderr_thread.join()
                    
                    return_code = process.wait()

                success = return_code == 0
                
                # 4. Find all the output files if the run was successful
                if success:
                    output_files = glob.glob(f'output/{outstring}_*.txt')

                return success, return_code, output_files
                
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Encore executable not found at '{self.encore_executable_path}'. "
                    "Please ensure it is compiled and in your system's PATH, or "
                    "provide the full path during initialization."
                )
            except Exception as e:
                return False, 1, []
            
        finally:
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)

# --- Example Usage ---
if __name__ == "__main__":
    wrapper = EncoreWrapper()
    
    print("--- Using a NumPy array with specific parameters ---")
    mock_data = np.random.rand(100, 4) * 100
    success, return_code, output_files = wrapper.run_correlation(
        input_data=mock_data, 
        rmin=10.0, 
        rmax=50.0, 
        nside=20, 
        outstring='my_np_test'
    )
    
    if success:
        print("Encore ran successfully.")
        print(f"Generated files: {output_files}")
    else:
        print(f"Encore failed with return code {return_code}.")
        
    print("\n" + "="*50 + "\n")

    print("--- Running in GPU mode with a mock FITS file ---")
    if fits:
        # Create a mock FITS file with different column names
        mock_fits_file = 'mock_fits_data.fits'
        col1 = fits.Column(name='RA', format='E', array=np.random.rand(50))
        col2 = fits.Column(name='DEC', format='E', array=np.random.rand(50))
        col3 = fits.Column(name='Z', format='E', array=np.random.rand(50))
        col4 = fits.Column(name='WEIGHT', format='E', array=np.random.rand(50))
        cols = fits.ColDefs([col1, col2, col3, col4])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(mock_fits_file, overwrite=True)
        
        # Run encore with the FITS file path and specify the column names
        # Also enable GPU mode 1 and the -2pcf flag.
        success, return_code, output_files = wrapper.run_correlation(
            input_data=mock_fits_file,
            column_names=['RA', 'DEC', 'Z', 'WEIGHT'],
            outstring='my_gpu_test',
            gpu_mode=1,
            only_2pcf=True
        )
        
        if success:
            print("Encore ran successfully with FITS file in GPU mode.")
            print(f"Generated files: {output_files}")
        else:
            print(f"Encore failed with return code {return_code}.")
            
        os.remove(mock_fits_file)

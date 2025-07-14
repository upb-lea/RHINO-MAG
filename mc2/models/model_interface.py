from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class ModelInterface(ABC):

    @abstractmethod
    def __call__(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: float,
    ) -> npt.NDArray[np.float64]:
        """Model prediction interface according to the challenge description.

        Args:
            B_past (np.array): The physical (non-normalized) flux density values from time
                step t0 to t1 with shape (past_sequence_length,)
            H_past (np.array): The physical (non-normalized) field values from time step
                t0 to t1 with shape (past_sequence_length,)
            B_future (np.array): The physical (non-normalized) flux density values from
                time step t1 to t2 with shape (future_sequence_length,)
            temperature (float): The temperature of the material as a scalar

        Returns:
            H_future (np.array): The physical (non-normalized) field values from time
                step t1 to t2 with shape (future_sequence_length,)
        """
        pass

    @abstractmethod
    def batched_prediction(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Model prediction interface for batched inputs, i.e. for inputs with an extra
        leading dimension.

        Args:
            B_past (np.array): The physical (non-normalized) flux density values from time
                step t0 to t1 with shape (n_batches, past_sequence_length)
            H_past (np.array): The physical (non-normalized) field values from time step
                t0 to t1 with shape (n_batches, past_sequence_length)
            B_future (np.array): The physical (non-normalized) flux density values from
                time step t1 to t2 with shape (n_batches, past_sequence_length)
            temperature (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_future (np.array): The physical (non-normalized) field values from time
                step t1 to t2 with shape (n_batches, past_sequence_length)
        """
        pass

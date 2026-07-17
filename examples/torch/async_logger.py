import time

import torch

from thunder.utils import AsyncLogger, Scalar, TensorBoardLogger, Workspace

if __name__ == "__main__":
    workspace = Workspace("./logs", "thunder/tests", run_name="async_logger")
    tb_logger = TensorBoardLogger(workspace)

    logger = AsyncLogger([tb_logger])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training...")
    for i in range(100):
        data = torch.randn(100, 100, device=device)
        loss = data.mean()
        logger.log({"loss": Scalar(loss.detach())}, step=i)
        time.sleep(0.01)
    logger.close()
    print("Done")

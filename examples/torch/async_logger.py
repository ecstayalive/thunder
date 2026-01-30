import time

from thunder.utils.torch import *

if __name__ == "__main__":
    tb_logger = TensorBoardLogger(log_dir="./logs/thunder/tests")

    logger = AsyncLogger([tb_logger])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training...")
    for i in range(100):
        data = torch.randn(100, 100, device=device)
        loss = data.mean()
        logger.log({"loss": loss}, step=i)
        time.sleep(0.01)
    logger.close()
    print("Done")

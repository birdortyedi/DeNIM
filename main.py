import fire

from engine import AWBCorrectionTrainer, AWBCorrectionTester


def run(
    cfg: str,
    is_train: bool = True
):  
    if is_train:
        runner = AWBCorrectionTrainer(cfg)
    else:
        runner = AWBCorrectionTester(cfg)
    runner()


if __name__ == "__main__":
    fire.Fire(run)

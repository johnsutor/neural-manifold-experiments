import hydra
from omegaconf import OmegaConf

from train import ModelTrainer


@hydra.main(version_base=None, config_path="conf/", config_name="train")
def main(cfg: OmegaConf):
    trainer = ModelTrainer(cfg)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()

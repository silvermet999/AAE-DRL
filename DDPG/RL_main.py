import torch
import torch.nn.parallel
from RL_trainer import Trainer
# from RL_tester import Tester

from AAE import AAE_archi_opt


def main():
    enc_gen = AAE_archi_opt.EncoderGenerator(27, )
    """-----------------------------------------------Data Loader----------------------------------------------------"""

    train_loader = AAE_archi_opt.dataloader
    valid_loader = AAE_archi_opt.val_dl

    # Test datasets
    # test_loader = AAE_archi_opt.test_dl

    """----------------Model Settings-----------------------------------------------"""
    de_model = AAE_archi_opt.Decoder(12, )
    d_model = AAE_archi_opt.Discriminator(12, )
    d_model.eval()

    trainer = Trainer(train_loader, valid_loader, enc_gen, d_model, de_model)
    trainer.train()
    # else:
    #     tester = Tester(test_loader, enc_gen, d_model)
    #     evaluater = tester.evaluate()
    #     for i in range(100):
    #         next(evaluater)


if __name__ == '__main__':
    main()

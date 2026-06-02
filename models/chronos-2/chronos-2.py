from ..basemodel import BaseModel


class ChronosModel(BaseModel):
    def __init__(self,
                 input_length: int,
                 target_length: int,
                 loss_function: str,
                 **kwargs):
        super(ChronosModel, self).__init__(
            input_dim=input_length,
            loss_function=loss_function,
            **kwargs
        )
        print("initialize Chronos model")
        self.save_hyperparameters()
        self.input_length = input_length
        self.target_length = target_length
        self._init_model()

    def _init_model(self):
        pass

    def forward(self, batch):
        x, *_ = batch
        y_pred = None
        return y_pred

    def configure_optimizers(self):
        pass

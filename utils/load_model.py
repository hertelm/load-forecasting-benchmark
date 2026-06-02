# TODO: instantiate model via config and therefore get rid of this function


def load_model(config):
    """
    Load the appropriate model based on the configuration.

    Arguments:
        config: hydra configuration object.

    Returns:
        An instantiated model based on the model_name in the config.
    """

    # Common model parameters
    common_params = {
        "loss_function": config.model.loss_function,
        "quantiles": config.model.quantiles,
        "window_size": config.sample.window_size,
    }

    # if config.model.model_name == "timexer":
    if config.model.model_class.startswith("timexer"):
        from models.timexer.timexer import TimexerModel

        static_features = True if (config.features.stat or config.features.static_time_dict) else False

        return TimexerModel(
            seq_len=config.sample.input_length,
            pred_len=config.sample.target_length,
            feature_lookback=config.sample.input_length,
            feature_lookahead=config.sample.target_length,
            static_features=static_features,
            patch_len=config.model.patch_len,
            use_norm=config.model.use_norm,
            n_vars=config.model.n_vars,
            dropout=config.model.dropout,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            activation=config.model.activation,
            e_layers=config.model.num_layers,
            optimizer=config.model.optimizer,
            lr=config.model.lr,
            lr_scheduler=config.model.lr_scheduler,
            torch_attention=config.model.torch_attention,
            **common_params,
        )  

    if config.model.model_class == "dlinear":
        from models.dlinear.dlinear import DlinearModel

        return DlinearModel(
            seq_len=config.sample.input_length,
            pred_len=config.sample.target_length,
            kernel_size=config.model.kernel_size,
            optimizer=config.model.optimizer,
            lr=config.model.lr,
            lr_scheduler=config.model.lr_scheduler,
            **common_params,
        )  

    if config.model.model_class.startswith("nhits"):
        from models.nhits.nhits import NHITS

        return NHITS(
            seq_len=config.sample.input_length,
            pred_len=config.sample.target_length,
            futr_feature_size=config.features.futr_feature_size,
            hist_feature_size=config.features.hist_feature_size,
            stat_feature_size=config.features.stat_feature_size,
            feature_lookback=config.sample.input_length,
            feature_lookahead=config.sample.target_length,
            optimizer=config.model.optimizer,
            lr=config.model.lr,
            lr_scheduler=config.model.lr_scheduler,
            stack_types=config.model.stack_types,
            n_blocks=config.model.n_blocks,
            mlp_units=config.model.mlp_units,
            n_pool_kernel_size=config.model.n_pool_kernel_size,
            n_freq_downsample=config.model.n_freq_downsample,
            interpolation_mode=config.model.interpolation_mode,
            dropout_prob_theta=config.model.dropout_prob_theta,
            activation=config.model.activation,
            **common_params,
        )
    if config.model.model_class.startswith("transformer"):
        from models.transformer.transformer import Transformer

        return Transformer(
            input_length=config.sample.input_length,
            target_length=config.sample.target_length,
            architecture=config.model.architecture,
            past_feature_size=config.features.hist_feature_size,
            future_feature_size=config.features.futr_feature_size,
            static_feature_size=config.features.stat_feature_size,
            patch_size=config.model.patch_size,
            conv_kernel_size=config.model.conv_kernel_size,
            conv_max_pooling=config.model.conv_max_pooling,
            lstm_layers=config.model.lstm_layers,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            attention=config.model.attention,
            max_pooling=config.model.max_pooling,
            dense_units=config.model.dense_units,
            num_dense_layers=config.model.num_dense_layers,
            dropout=config.model.dropout,
            optimizer=config.model.optimizer,
            lr=config.model.lr,
            lr_scheduler=config.model.lr_scheduler,
            **common_params,
        )
    if config.model.model_class.startswith("tft"):
        from models.TFT.tft import TFT

        return TFT(
            h=config.sample.target_length,
            input_size=config.sample.input_length,
            past_feature_size=config.features.hist_feature_size,
            future_feature_size=config.features.futr_feature_size,
            static_size=config.features.stat_feature_size,
            hidden_size=config.model.hidden_size,
            n_head=config.model.n_head,
            attn_dropout=config.model.attn_dropout,
            dropout=config.model.dropout,
            learning_rate=config.model.lr,
            **common_params,
        )
    if config.model.model_class == "naive":
        from baselines.naive.naive_model import NaiveQuantileModel
        
        return NaiveQuantileModel(
            quantiles=config.model.quantiles,
            naive_input_length=config.sample.window_size,
            target_length=config.sample.target_length,
        )
    else:
        raise NotImplementedError(f"Model {config.model.model_name} not implemented.")

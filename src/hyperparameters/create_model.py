from ..models import UnsupervisedAE, UnsupervisedVAE, SupervisedAE, SupervisedVAE

def create_model(model_class, input_dim, latent_dim, activation, dropout_rate, kl_weight=None, classifier_dims=None):
    if model_class in [UnsupervisedVAE, SupervisedVAE]:
        return model_class(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            kl_weight=kl_weight if kl_weight is not None else 1.0
        )
    elif model_class == SupervisedAE:
        return model_class(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            latent_dim=latent_dim,
            classifier_dims=classifier_dims,  # Einzelne Schicht für SAE
            activation=activation,
            dropout_rate=dropout_rate
        )
    elif model_class == UnsupervisedAE:  
        return model_class(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )
    else:
        raise Exception("unknown model class {model_class}")
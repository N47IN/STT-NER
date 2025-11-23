from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


from transformers import AutoConfig

def create_model(model_name: str, classifier_dropout: float = 0.2, label_smoothing: float = 0.1):
    """
    Create token classification model with configurable dropout and label smoothing.
    
    Args:
        model_name: HuggingFace model name
        classifier_dropout: Dropout probability for classifier layer (helps prevent overfitting)
        label_smoothing: Label smoothing factor (prevents overconfidence, 0.0 = no smoothing)
    """
    # Load config first and modify dropout settings
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    
    # Set dropout rates if supported by the model
    if hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = 0.2  # Increase dropout for regularization
    if hasattr(config, 'attention_probs_dropout_prob'):
        config.attention_probs_dropout_prob = 0.2
    if hasattr(config, 'dropout'):
        config.dropout = 0.2
    if hasattr(config, 'seq_classif_dropout'):
        config.seq_classif_dropout = classifier_dropout
    
    # Create model with modified config
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config
    )
    
    return model

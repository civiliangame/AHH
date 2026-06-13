"""Configuration for running Kintsugi Depression and Anxiety model."""

import torch

EXPECTED_SAMPLE_RATE = 16000 # Audio sample rate in hertz

# Configuration for running Kintsugi Depression and Anxiety model as intended
default_config = {
    # See featex.py for preprocessor config details
    'preprocessor_config': {
        'normalize_features': True,
        'chunk_seconds': 30,
        'max_overlap_frac': 0.0,
        'pad_last_chunk_to_full': True,
    },

    # See model.py for backbone config details
    'backbone_configs': {'audio': {'model': 'openai/whisper-small.en',
                                   'hf_config': {'encoder_layerdrop': 0.0,
                                                 'dropout': 0.0,
                                                 'activation_dropout': 0.0},
                                   'lora_params': {'r': 32,
                                                   'lora_alpha': 64.0,
                                                   'target_modules': 'all-linear',
                                                   'lora_dropout': 0.4,
                                                   'modules_to_save': ['conv1', 'conv2'],
                                                   'bias': 'all'}},
                         'llma': {'model': 'openai/whisper-small.en',
                                  'hf_config': {'encoder_layerdrop': 0.0,
                                                'dropout': 0.0,
                                                'activation_dropout': 0.0}}},

    # See model.py for classifier config details
    'classifier_config': {'shared_projection_dim': [256, 64],
                          'tasks': {'depression': {'proj_dim': 128, 'dropout': 0.4},
                                    'anxiety': {'proj_dim': 128, 'dropout': 0.4}}},

    # Score thresholds chosen to optimize macro average F1 score on validation set
    'inference_thresholds': {
        # Three-level depression severity model:
        #             depression score <= -0.6699 --> no depression (PHQ-9 <= 9)
        #   -0.6699 < depression score <= -0.2908 --> mild to moderate depression (10 <= PHQ-9 <= 14)
        #   -0.2908 < depression score            --> severe depression (PHQ-9 >= 15)
        'depression': [-0.6699, -0.2908],
        # Four-level anxiety severity model:
        #             anxiety score <= -0.7939 --> no anxiety (GAD-7 <= 4)
        #   -0.7939 < anxiety score <= -0.2173 --> mild anxiety (5 <= GAD-7 <= 9)
        #   -0.2173 < anxiety score <=  0.1521 --> moderate anxiety (10 <= GAD-7 <= 14)
        #    0.1521 < anxiety score            --> severe anxiety (GAD-7 >= 15)
        'anxiety': [-0.7939, -0.2173, 0.1521]
    }
}

# Average filter bank energies used for feature normalization
logmel_energies = torch.tensor([0.34912264, 0.58558977, 0.7912451 , 0.92767584, 0.98273695,
       0.98439455, 0.9603633 , 0.93906444, 0.9366281 , 0.93200225,
       0.916437  , 0.8928787 , 0.8637211 , 0.83265126, 0.79977655,
       0.7778334 , 0.7561299 , 0.72997606, 0.70391226, 0.6800474 ,
       0.65755   , 0.63536274, 0.61355984, 0.5923383 , 0.5720056 ,
       0.55244887, 0.53684795, 0.5221597 , 0.5098636 , 0.49923953,
       0.48908615, 0.47840047, 0.46758702, 0.47343993, 0.46268672,
       0.4475126 , 0.46747103, 0.45131385, 0.4635319 , 0.44889897,
       0.45491976, 0.4373785 , 0.43154317, 0.42194438, 0.41158468,
       0.40096927, 0.3933149 , 0.38795966, 0.38441542, 0.38454026,
       0.3815766 , 0.3768835 , 0.3719921 , 0.3654539 , 0.35399568,
       0.3425986 , 0.32823247, 0.31404305, 0.30564603, 0.29617435,
       0.29273877, 0.28560263, 0.27459458, 0.26876706, 0.25825337,
       0.24759005, 0.24090728, 0.2344712 , 0.22529823, 0.20880115,
       0.193578  , 0.18290243, 0.17621627, 0.17087021, 0.16641389,
       0.15932252, 0.14312662, 0.11790597, 0.08030523, 0.03747071],
)
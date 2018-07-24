import collections


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'merge_method',
        'add_image_level_feature',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant'
    ])):
    """Immutable class to hold model options."""

    __slots__ = ()

    def __new__(cls,
                outputs_to_num_classes,
                crop_size=None,
                atrous_rates=None,
                output_stride=8,
                merge_method='max',
                add_image_level_feature=True,
                aspp_with_batch_norm=True,
                aspp_with_separable_conv=True,
                multi_grid=None,
                decoder_use_separable_conv=True,
                logits_kernel_size=1,
                decoder_output_stride=4,
                model_variant='xception_65'):
        """Constructor to set default values.

        Args:
          outputs_to_num_classes: A dictionary from output type to the number of
            classes. For example, for the task of semantic segmentation with 21
            semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
          crop_size: A tuple [crop_height, crop_width].
          atrous_rates: A list of atrous convolution rates for ASPP.
          output_stride: The ratio of input to output spatial resolution.

        Returns:
          A new ModelOptions instance.
        """
        return super(ModelOptions, cls).__new__(
            cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
            merge_method, add_image_level_feature,
            aspp_with_batch_norm, aspp_with_separable_conv,
            multi_grid, decoder_output_stride,
            decoder_use_separable_conv, logits_kernel_size,
            model_variant)

import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, num_gaze_outputs=2):
        """
        Initializes the Simple Swin Gaze Model.

        Args:
            model_name (str): The name of the Swin Transformer model to load from timm.
            pretrained (bool): Whether to load pretrained weights (ImageNet).
            num_gaze_outputs (int): The number of output values for gaze (e.g., 2 for pitch and yaw).
        """
        super().__init__()
        
        # Load the specified Swin Transformer model using timm
        # By setting num_classes, timm will automatically replace the original classification head
        # with a new nn.Linear layer suited for 'num_gaze_outputs'.
        # If pretrained=True, the backbone weights are loaded, and the new head is usually randomly initialized.
        self.swin_transformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_gaze_outputs  # This adapts the head for our 2 gaze outputs
        )

        self.loss_op = nn.L1Loss()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
                               The image should be normalized as expected by the Swin model.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_gaze_outputs) for pitch and yaw.
        """
        return self.swin_transformer(x["face"])

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label)
        return loss

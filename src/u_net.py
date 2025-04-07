import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class SimpleUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
   
        base_channels = 32
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        
        # Decoder (upsampling path)
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec1 = DoubleConv(base_channels * 8, base_channels * 4)  # Note: 8 -> 4 because of skip connection
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)  # Note: 4 -> 2 because of skip connection
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.dec3 = DoubleConv(base_channels * 2, base_channels)  # Note: 2 -> 1 because of skip connection
        
        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder 1
        enc1_out = self.enc1(x)
        enc2_in = self.down1(enc1_out)

        # Encoder 2
        enc2_out = self.enc2(enc2_in)
        enc3_in = self.down2(enc2_out)

        # Encoder 3
        enc3_out = self.enc3(enc3_in)
        bottleneck_in = self.down3(enc3_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)

        # Decoder 1
        dec1_in = self.up1(bottleneck_out)
        dec1_in = torch.cat([dec1_in, enc3_out], dim=1)
        dec1_out = self.dec1(dec1_in)

        # Decoder 2
        dec2_in = self.up2(dec1_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)

        # Decoder 3
        dec3_in = self.up3(dec2_out)
        dec3_in = torch.cat([dec3_in, enc1_out], dim=1)
        dec3_out = self.dec3(dec3_in)

        # Output layer
        output = self.out_conv(dec3_out)
        return output

    def forward_with_print(self, x):
        # Encoder
        print(x.shape)
        enc1_out = self.enc1(x)
        print("After enc1")
        print(enc1_out.shape)
        enc2_in = self.down1(enc1_out)
        print("After down1")
        print(enc2_in.shape)
        enc2_out = self.enc2(enc2_in)
        print("After enc2")
        print(enc2_out.shape)
        enc3_in = self.down2(enc2_out)
        print("After down2")
        print(enc3_in.shape)
        enc3_out = self.enc3(enc3_in)
        print("After enc3")
        print(enc3_out.shape)
        bottleneck_in = self.down3(enc3_out)
        print("After down3")
        print(bottleneck_in.shape)
        # Bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)
        print("After bottleneck")
        print(bottleneck_out.shape)
        # Decoder
        dec1_in = self.up1(bottleneck_out)
        print("After up1")
        print(dec1_in.shape)
        # Skip connection - concatenate encoder output with decoder input
        dec1_in = torch.cat([dec1_in, enc3_out], dim=1)
        print("After cat")
        print(dec1_in.shape)
        dec1_out = self.dec1(dec1_in)
        print("After dec1")
        print(dec1_out.shape)
        dec2_in = self.up2(dec1_out)
        print("After up2")
        print(dec2_in.shape)
        # Skip connection
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        print("After cat2")
        print(dec2_in.shape)
        dec2_out = self.dec2(dec2_in)
        print("After dec2")
        print(dec2_out.shape)
        dec3_in = self.up3(dec2_out)
        print("After up3")
        print(dec3_in.shape)
        # Skip connection
        dec3_in = torch.cat([dec3_in, enc1_out], dim=1)
        print("After cat3")
        print(dec3_in.shape)
        dec3_out = self.dec3(dec3_in)
        print("After dec3")
        print(dec3_out.shape)
        # Output layer
        output = self.out_conv(dec3_out)
        print("After out_conv")
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    x = torch.randn(4, 1, 64, 64)
    model = SimpleUnet(in_channels=1, out_channels=1)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    output = model.forward_with_print(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
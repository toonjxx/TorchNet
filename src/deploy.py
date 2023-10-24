import torch
import argparse
from models.convnext import ConvNeXt
from utils import load_sample
import numpy as np

def main():
    # Add parser
    parser = argparse.ArgumentParser()

    # Add parser for device
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')

    # Add parser for model
    parser.add_argument('--model', type=str, default='ConvNeXt1d_S_Hypertension', help='model name')

    # Add parser for checkpoint
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')

    # Add parser for sample
    parser.add_argument('--samplepath', type=str, default='sample.csv', help='sample file path')

    # Parse arguments
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[32, 64, 128, 256],in_chans=1,num_classes=1)
    model.load_state_dict(torch.load(args.checkpoint,map_location='cuda'))
    model.eval()
    model.to(device)
    sample = load_sample(args.samplepath)

    # Check model name contains 'Hypertenstion' or 'BP'
    if (True):
        if sample is not None:
            sample = torch.FloatTensor(sample.values).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(sample)
                print(output)
                output = torch.sigmoid(output)
                print(output)
                output = torch.round(output)
                print(output)
            if output == 0:
                print('No Hypertension')
            else:
                print('Hypertension')
        else:
            print('Sample is None')
    elif (args.model.contains('BP')):
        sample = torch.tensor(sample).to(device)
        predict = model(sample)
        predict = predict*170
        print(f'Predicted SBP: {predict}')
    

if __name__ == '__main__':
    main()
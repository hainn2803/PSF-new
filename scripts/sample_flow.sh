model_path="output/train_flow/2024-10-31-19-56-18/epoch_1.pth"
dataroot="datasets/ShapeNetCore.v2.PC15k/"
category="airplane"

workers=4
dist="single"

epoch=200
batch_size=32

CUDA_VISIBLE_DEVICES=2 python3 sample_flow.py --model "$model_path" \
              	      --category "$category" \
                      --dataroot "$dataroot" \
                      --workers "$workers" \
                      --nEpochs "$epoch" \
		              --batch_size "$batch_size" \
                      --distribution_type "$dist"
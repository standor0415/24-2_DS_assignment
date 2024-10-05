from darknet import Darknet
import torch
import torch.nn as nn

# Darknet 네트워크 구조 설정
architecture_config = [
    (7, 64, 2, 3),  # (kernel size, number of filters, stride, padding)
    "M",  # max-pooling 2x2 stride = 2
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [(conv1), (conv2), repeat times]
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# YOLOv1 모델 정의
class Yolov1(nn.Module):
    def __init__(self, split_size, num_boxes, num_classes):
        super(Yolov1, self).__init__()
        self.darknet = Darknet(architecture_config)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        # 구현한 darknet과 fully connected layer로 yolo의 layer를 구성해주세요!
        # 텐서 평탄화도 잊지 않기~
        
        ## TODO
        x = self.darknet(x)
        x = x.view(-1, 1024*7*7)
        x = self.fcs(x)
        
        return x

    def _create_fcs(self, split_size, num_boxes, num_classes, dropout=0.5, leakyReLU=0.1):
        S, B, C = split_size, num_boxes, num_classes
        final_cnn_out_channel = 1024
        layer_output = 4096

        # Fully connected layer 수정
        linear1 = nn.Linear(in_features=S * S * final_cnn_out_channel, out_features=layer_output)  # 1024 * 7 * 7 을 계산
        linear2 = nn.Linear(in_features=layer_output, out_features=S * S * (C + B * 5))
        drop = nn.Dropout(p=dropout)
        leak_relu = nn.LeakyReLU(leakyReLU)

        return nn.Sequential(linear1, drop, leak_relu, linear2)


def test():

    split_size = 7  # S (그리드 크기)
    num_boxes = 2   # B (각 셀이 예측하는 바운딩 박스 수)
    num_classes = 20  # C (클래스 수)

    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)

    # 더미 입력 데이터 생성 (batch_size: 1, channel: 3, height: 448, width: 448)
    dummy_input = torch.randn(1, 3, 448, 448)

    # 모델에 데이터 전달 및 출력 확인
    output = model(dummy_input)
    
    expected_shape = (1, split_size * split_size * (num_classes + num_boxes * 5))
    print(f"Expected shape: {expected_shape}")
    print("Test passed!" if output.shape == expected_shape else "Test failed!")


if __name__ == "__main__":
    test()
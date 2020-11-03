import time
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


epochs = 10
batchsz = 2048
lr = 1e-3


cifar_train = datasets.CIFAR10('cifar10', train=False, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]), download=False)

cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True) 

cifar_test = datasets.CIFAR10('cifar10', train=False, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]), download=False)

cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

trained_model = resnet18(pretrained=True)
model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                        Flatten(), # [b, 512, 1, 1] => [b, 512]
                        nn.Linear(512, 10))

# 验证是否有多个gpu，如果有就输出使用了多少个gpu
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

model = model.to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def main():
    since = time.time()
    global_step = 0
    for epoch in range(epochs):
        print('-' * 15, "epoch {}".format(epoch), '-'*15)
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32], [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if batchidx % 10 == 0:
                print('  batch {}'.format(batchidx), 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32], [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)
            
    time_elapsed = time.time() - since
	# 输出花费的总时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()

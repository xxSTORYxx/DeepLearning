import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

# Hyper Parameters, GAN is Sensitive to it!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR_D = 0.001 # 推薦 adam 配 lr=3e-4
LR_G = 0.001
Z_DIM = 32
IMG_DIM = 1 * 28 * 28
BATCH_SIZE = 64
EPOCH = 200

disc = Discriminator(IMG_DIM).to(device)
gen = Generator(Z_DIM, IMG_DIM).to(device)
fixed_noise = torch.randn(size=(BATCH_SIZE, Z_DIM)).to(device)
# transforms = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))] 
# ) # mnist 官方提供的 normalize 均值與標準差 0.1307, 0.3081,但仍要 normalize 0.5, 0.5 詳見評論區
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))] 
)
dataset = datasets.MNIST(
    root = "./mnist/",
    transform = transforms,
    download = False
)
loader = DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

print(f'train dataset size = {len(dataset)}')
opt_disc = optim.Adam(disc.parameters(), lr = LR_D)
opt_gen = optim.Adam(gen.parameters(), lr = LR_G)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f'./runs_gan/GAN_MNIST/fake')
writer_real = SummaryWriter(f'./runs_gan/GAN_MNIST/real')
step = 0

for epoch in range(EPOCH):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(real.shape[0], -1).to(device)
        # print(f'real.shape={real.shape}') # 32 x 784
        batch_size = real.shape[0]

        # train discriminator loss : max log(D(real)) + log (1 - D(G(z)))
        noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)
        fake =gen(noise) # generator 生成假資料
        disc_real = disc(real).view(-1) # 變成 1 維
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # log(D(real))
        # 跟 disc_real 形狀相同的 1 矩陣, y_n, 雖然 BCE loss 前有負號, 但因為 max + = min - 所以不用改
        disc_fake = disc(fake).view(-1)
        # disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # log(1 - D(G(z)))
        # 跟 disc_real 形狀相同的 0 矩陣, y_n, 雖然 BCE loss 前有負號, 但因為 max + = min - 所以不用改
        lossD = (lossD_real + lossD_fake) / 2
        opt_disc.zero_grad()
        # lossD.backward()
        lossD.backward(retain_graph = True) # 替代 detach() 的作法, 因為 backward() 會把所有資料清空
        opt_disc.step()


        # train generator loss : min log(1 - D(z)) <-> max log(D(G(z)))
        output = disc(fake).view(-1) #把fake送進更新完的disc可以得到更好的output
        lossG = criterion(output, torch.ones_like(output))
        # lossG = criterion(disc_fake, torch.ones_like(disc_fake))
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        if batch_idx == 0:
            print(
                f'Epoch [{epoch:03d}/{EPOCH+1}] \ '
                f'Loss D : {lossD:.4f}, Loss G : {lossG:.4f}'
            )
            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28) 
                # real 此時已經被壓成 32 x 784, 所以要用 data, 因為 reshape 不會動到 real
                # print(f'data.shape={data.shape}')
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step = step
                )
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step = step
                )
                step += 1




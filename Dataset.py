import torchvision.transforms as transforms
import torchvision
import torch


class MNISTDataset:
    def __init__(self, train=True):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.data = torchvision.datasets.MNIST(root="./MNIST", train=train, download=True, transform=transform)

    def preprocess_data(self, data_set):
        x = torch.flatten(data_set.data.type(torch.FloatTensor), start_dim=1)
        y = data_set.targets
        x /= 255.0
        return x, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_data(self):

        self.x, self.y = self.preprocess_data(self.data)

        return self.x, self.y

    def view_one_image(self, orig_image_idx):

        # print(orig_image_idx)

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        orig_image, label = self.data[orig_image_idx]

        orig_image = orig_image.numpy().squeeze()

        axs.imshow(orig_image, cmap='gray')
        axs.set_title(f'{label}')
        axs.axis('off')

        plt.show()



    def save_image(self, orig_image_idx, origin_dense, origin_sparse, random_img, dense_pred, sparse_pred, file_name):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        orig_image, label = self.data[orig_image_idx]

        orig_image = orig_image.numpy().squeeze()

        axs[0].imshow(orig_image, cmap='gray')
        axs[0].set_title(f'Dense predict: {int(origin_dense)} - Sparse predict: {int(origin_sparse)}')
        axs[0].axis('off')

        reshaped_random_image = torch.tensor(random_img).view(28, 28).numpy()
        axs[1].imshow(reshaped_random_image, cmap='gray')
        axs[1].set_title(f'Dense predict: {int(dense_pred)} - Sparse predict: {int(sparse_pred)}')
        axs[1].axis('off')

        # Convert the tensor image to a numpy array for visualization
        # image = image.numpy().squeeze()

        plt.savefig(file_name)

        # plt.show()

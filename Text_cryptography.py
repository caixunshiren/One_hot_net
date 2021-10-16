import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.cuda.empty_cache()
import matplotlib.pyplot as plt


def uniform_initializer(out_dim, in_dim, cuda = True):
    tensor = torch.empty(out_dim, in_dim)
    if cuda:
        return torch.nn.init.uniform_(tensor, a=-1, b=1).cuda()
    else:
        return torch.nn.init.uniform_(tensor, a=-1, b=1)


# noise
def apply_gaussian_noise(tensor, sd, device=torch.device("cuda:0")):
    tensor = tensor + (sd) * torch.randn(*tuple(tensor.shape)).to(device)
    return tensor


from sklearn.preprocessing import MinMaxScaler


class naive_crossbar():
    def __init__(self, R_on, R_off, viability, P_stuck_on, P_stuck_off, device=torch.device("cuda:0")):
        self.Gon = 1 / R_on
        self.Goff = 1 / R_off
        self.viability = viability
        self.P_stuck_on = P_stuck_on
        self.P_stuck_off = P_stuck_off
        self.device = device

    def stuck_filter(self, shape, P):
        return ((torch.rand(shape) > (1 - P)) * 1.0).to(self.device)

    def apply_filter(self, W, f, weight):
        shape = W.shape
        inv = torch.ones(shape).to(self.device) - f
        return W * inv + weight * f

    def convert(self, W, viability=None):
        if viability is None:
            viability = self.viability
        shape = W.shape
        # get stuck_on filter
        stuck_on_filter = self.stuck_filter(shape, self.P_stuck_on)
        # get stuck_off filter
        stuck_off_filter = self.stuck_filter(shape, self.P_stuck_off)

        # scaler transform
        scaler = MinMaxScaler(feature_range=(self.Goff, self.Gon))
        W = scaler.fit_transform(W.reshape(-1, 1).cpu()).reshape(shape)
        W = torch.tensor(W, dtype=torch.float, device=self.device)

        # add Viability
        W = apply_gaussian_noise(W, viability * (self.Gon - self.Goff))

        # add filters
        W = self.apply_filter(W, stuck_on_filter, self.Gon)
        W = self.apply_filter(W, stuck_off_filter, self.Goff)

        # clip
        W = torch.clip(W, min=self.Goff, max=self.Gon)
        W = scaler.inverse_transform(W.cpu())
        return torch.tensor(W, dtype=torch.float, device=self.device)


class simple_encoder_wthreshold():
    def __init__(self, out_dim, in_dim, epsilon, crossbar_params, cuda=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = uniform_initializer(out_dim, in_dim, cuda)
        self.device = torch.device("cuda:0") if cuda else torch.device("cpu")
        self.epsilon = epsilon
        self.cb = naive_crossbar(crossbar_params['Ron'], crossbar_params['Roff'], crossbar_params['viability'], crossbar_params['P_stuck_on'], crossbar_params['P_stuck_off'])

    def apply(self, X):
        return (torch.matmul(self.W, X) > self.epsilon).float()

    def apply_wnoise(self, X, sd):
        # print(X.shape, self.W.shape)
        if sd == 0:
            return self.apply(X)
        # return (torch.matmul(apply_gaussian_noise(self.W, sd, device = self.device), X) > self.epsilon).float()
        return (torch.matmul(self.cb.convert(self.W, sd), X) > self.epsilon).float()

    def apply_wnoise_realistic(self, X, sd):
        # print(X.shape, self.W.shape)
        if sd == 0:
            return self.apply(X)
        encoded = torch.zeros((self.out_dim, X.shape[1])).to(self.device)
        for i in range(X.shape[1]):
            # encoded[:,i] = (torch.matmul(apply_gaussian_noise(self.W, sd, device = self.device), X[:,i].view(-1,1)) > self.epsilon).float().view(-1)
            encoded[:, i] = (
                        torch.matmul(self.cb.convert(self.W, sd), X[:, i].view(-1, 1)) > self.epsilon).float().view(-1)
        return encoded


class One_hot_net(nn.Module):
    def __init__(self, in_dim, n_class, f_encoder, encoder_multiplier, f_initializer, epsilon):
        super(One_hot_net, self).__init__()
        self.in_dim = in_dim
        feature_len = in_dim * encoder_multiplier
        self.feature_len = feature_len
        self.epsilon = epsilon
        self.n_class = n_class
        self.f_encoder = f_encoder
        self.f_initializer = f_initializer
        self.tail = nn.Linear(feature_len, n_class)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = self.f_encoder.apply_wnoise(X, noise)
        X = torch.transpose(X, 0, 1)
        X = self.tail(X)
        # print(X.shape)
        X = self.output(X)
        # print(X.shape)
        return X

    def decrypt(self, X):
        X = torch.transpose(X, 0, 1)
        X = self.tail(X)
        # print(X.shape)
        X = self.output(X)
        # print(X.shape)
        return X

def make_data(size, classes):
    data = torch.rand(size) * classes
    return data.long()#, F.one_hot(data.long(), num_classes=classes)

def encrypt(message, model, secret_key, n):
    message_index = [(ord(c)-32) for c in message]
    secret_message = secret_key[:, message_index]#.to(device)
    return model.f_encoder.apply_wnoise_realistic(secret_message, n)

def determine_letter(one_hots):
    ind = torch.argmax(one_hots, dim=1)
    m = ""
    for i in ind:
        m += chr(int(i)+32)
    return m

def decrypt(encrypted_message, model, secret_key):
    raw_m = model.decrypt(encrypted_message)
    m = determine_letter(raw_m)
    return m

def decryption_accuracy(message, decrypted_message):
    count = 0
    for i in range(len(message)):
        if message[i] == decrypted_message[i]:
            count+=1
    return count/len(message) * 100

def visualize(message, model, secret_key, n):
    encrypted_message = encrypt(message, model, secret_key, n = n)
    plt.imshow(encrypted_message.cpu().detach().numpy(),  cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()

class encryption_model():
    def __init__(self, model_name, secret_key, cryoto_params, crossbar_params):
        #initialization
        self.name = model_name
        self.secret_key = secret_key
        self.crypto_params = cryoto_params
        self.crossbar_params = crossbar_params
        self.model = None
        self.optimizer = None

    def train(self):
        secret_key =self.secret_key
        classes = self.crypto_params['classes']
        skdim = self.crypto_params['skdim']
        dim_multiplier = self.crypto_params['dim_multiplier']

        parameters = {
            'in_dim': skdim,
            'n_class': classes,
            'f_encoder': simple_encoder_wthreshold(skdim * dim_multiplier, skdim, 0, self.crossbar_params),
            'f_initializer': uniform_initializer,
            'encoder_multiplier': dim_multiplier,
            'epsilon': 0,
            'n_layers': 1,
            'layer_size_factor': [1],
            'dropout': [-1]
        }

        device = torch.device("cuda:0")
        model1 = One_hot_net(parameters['in_dim'], parameters['n_class'], parameters['f_encoder'],
                             parameters['encoder_multiplier'],
                             parameters['f_initializer'], parameters['epsilon']).to(device)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.5, momentum=0.5)

        n_epochs = 10
        n_batch = 2000
        batch_size_train = 50
        batch_size_test = 50

        train_loader = []
        test_loader = []
        for i in range(n_batch):
            # train
            indices = make_data(batch_size_train)
            train_loader.append((secret_key[:, indices], indices))

            # test
            indices = make_data(batch_size_train)
            test_loader.append((secret_key[:, indices], indices))
        print(train_loader[0][0].shape, train_loader[0][1].shape, train_loader[0][1])
        train_losses = []
        test_losses = []

        def train(epoch, model, optimizer, trainloader, log_interval=10, device=torch.device("cuda:0")):
            model.train()
            train_loss = 0
            for batch_idx, data_set in enumerate(train_loader):
                data = data_set[0].to(device)
                labels = data_set[1].to(device)
                optimizer.zero_grad()
                output = model(data)
                # print(output.shape, output)
                loss = F.nll_loss(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss
            train_loss /= len(train_loader)
            train_losses.append(float(train_loss))
            print('Epoch: {}, Train set: Avg. loss: {:.6f}'.format(epoch,
                                                                   train_loss))
            return model, optimizer

        def test(model, test_loader, device=torch.device("cuda:0")):
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data_set in test_loader:
                    data = data_set[0].to(device)
                    labels = data_set[1].to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, labels).item()
            test_loss /= len(test_loader)
            test_losses.append(float(test_loss))
            # print(output[:,0]-data[:,0])
            print('Test set: Avg. loss: {:.6f}'.format(
                test_loss))
            return test_loss

        for epoch in range(1, n_epochs + 1):
            train(epoch, model1, optimizer1, train_loader)
            test(model1, test_loader)

        plt.plot(range(len(train_losses)), train_losses, color='blue')
        plt.scatter(range(len(test_losses)), test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training epochs')
        plt.ylabel('MSE loss')
        plt.show()

        self.model = model1
        self.optimizer = optimizer1


    def evaluate(self, message, noise = None, vis = True):
        secret_key = self.secret_key
        noise = noise if noise is not None else self.crypto_params['noise']
        model1 = self.model
        encrypted_message = encrypt(message, model1, secret_key, n=0.4)
        decrypted_m = decrypt(encrypted_message, model1, secret_key)
        acc = decryption_accuracy(message, decrypted_m)
        print("decryption accuracy is {}%".format(round(acc, 2)))
        if vis:
            visualize("a" * 200 + "b" * 200 + "c" * 200 + "d" * 200 + "e" * 200, model1, secret_key, n=noise)
        return acc

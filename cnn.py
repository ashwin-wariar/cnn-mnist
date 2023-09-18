import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def im2col(A, size):
    H, W = A.shape
    steps_h = H - size + 1
    steps_w = W - size + 1
    im = []
    for h in range (steps_h):
        for w in range (steps_w):
            im.append(np.reshape(A[h:h+size, w:w+size], -1))
    im = np.transpose(np.asarray(im))

    return im


def get_mini_batch(im_train, label_train, batch_size):
    im, data = im_train.shape

    mini_batch_x = []
    mini_batch_y = []
    data_perm = np.random.permutation(data)
    batch = data // batch_size
    label_train = label_train.reshape(-1)
    label_train = label_train[data_perm]
    im_train = im_train[:, data_perm]
    classes = np.max(label_train) + 1
    #print(label_train.shape)
    #print(im_train.shape)

    for i in range(batch):
        mini_batch_x.append(im_train[:, i * batch_size : (i + 1) * batch_size])
        one_hot = np.eye(classes)[label_train[i * batch_size : (i + 1) * batch_size]]
        mini_batch_y.append(one_hot.T)
    
    
    mini_batch_x = np.asarray(mini_batch_x)
    mini_batch_y = np.asarray(mini_batch_y)
    #print(mini_batch_x.shape)
    #print(mini_batch_y.shape)
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    w_x = w @ x
    y = w_x + b
    #print(y)
    return y


def fc_backward(dl_dy, x, w, b):
    # print(dl_dy.shape)
    # print(w.shape)
    # dl_dx = dl_dy @ w
    # dl_dw = (x @ dl_dy).T
    # dl_db = dl_dy.T
    #print(dl_dy.shape, x.shape, w.shape, b.shape)
    dl_dy = dl_dy.reshape((-1, 1))
    dl_dx = (dl_dy.T) @ w
    dl_dw = dl_dy @ (x.T)
    dl_db = np.sum(dl_dy, axis=1).reshape((-1, 1))
    return dl_dx, dl_dw, dl_db


def relu(x):
    y = np.where(x > 0, x, 0)
    #print(y)
    return y


def relu_backward(dl_dy, x):
    #print(x.shape)
    dy_dx = np.where(x > 0, 1, 0)
    #print(dy_dx)
    dl_dx = (dl_dy.flatten() * dy_dx.flatten()).reshape(x.shape)
    #print(dl_dx.shape)
    return dl_dx


def loss_cross_entropy_softmax(x, y):
    x_t = np.exp(x)
    x_sum = np.sum(x_t)
    y_soft = x_t / (x_sum + 0.000000000001)
    l = - np.sum(y * np.log(y_soft))
    dl_dx = y_soft - y
    return l, dl_dx


def conv(x, w_conv, b_conv):
    H, W, C1 = x.shape
    h, w, C1, w_C2 = w_conv.shape
    #print(x.shape, w_conv.shape)
    y = np.zeros((H, W, w_C2))

    for i in range(C1):
        for j in range(w_C2):
            x_conv = np.pad(x[:,:,i], ((1,1), (1,1)))
            x_conv = im2col(x_conv, h)
            x_conv = np.transpose(x_conv)

            w2 = w_conv[:,:,i,j]
            w2 = np.reshape(w2, (-1,1))

            w_x = np.dot(x_conv, w2)
            w_x = w_x.reshape(H, W)

            y[:,:,j] += w_x + b_conv[j, 0]


    return y


def conv_backward(dl_dy, x, w_conv, b_conv):
    w_h, w_w, C1, C2 = w_conv.shape
    h, w, _ = dl_dy.shape
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    for i in range(C2):
        dy_db = dl_dy[:,:,i]
        dl_db[i,0] = np.sum(dy_db)

    for j in range(C2):
        dl_dy_ = dl_dy[:,:,j].reshape(-1)
        for k in range(C1):
            x_conv = np.pad(x[:,:,k], ((1,1), (1,1)))
            x_conv = im2col(x_conv, w_h)
            dy_dw = x_conv.T
            dl_dw[:,:,k,j] = np.dot(dl_dy_, dy_dw).reshape(w_h, w_w)
    return dl_dw, dl_db


def pool2x2(x):
    H, W, C = x.shape
    h_2 = int(H/2)
    w_2 = int(W/2)
    y = np.empty((h_2, w_2, C))

    stride = 2
    for c in range(C):
        for h in range(h_2):
            for w in range(w_2):
                y[h, w, c] = np.max(x[stride * h: stride * h + stride, stride * w: stride * w + stride, c])

    return y


def pool2x2_backward(dl_dy, x):
    h, w, c = dl_dy.shape
    dl_dx = np.zeros(x.shape)

    for i in range(c):
        for j in range(h):
            for k in range(w):
                x_back = x[j*2:j*2+2, k*2:k*2+2, i]
                mask = (x_back == np.max(x_back))
                dl_dx[j*2:j*2+2, k*2:k*2+2, i] = mask*dl_dy[j, k, i]

    return dl_dx


def flattening(x):
    return x.reshape((-1, 1))


def flattening_backward(dl_dy, x):
    return dl_dy.reshape(x.shape)


def train_mlp(mini_batch_x, mini_batch_y, learning_rate=0.1, decay_rate=0.95, num_iters=10000):
    nh = 30

    # initialization network weights
    w1 = np.random.randn(nh, 196)  # (30, 196)
    b1 = np.zeros((nh, 1))  # (30, 1)
    w2 = np.random.randn(10, nh)  # (10, 30)
    b2 = np.zeros((10, 1))  # (10, 1)

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training MLP'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dl_dw1_batch = np.zeros((nh, 196))
        dl_db1_batch = np.zeros((nh, 1))
        dl_dw2_batch = np.zeros((10, nh))
        dl_db2_batch = np.zeros((10, 1))
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros((batch_size, 1))
        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]]
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = fc(x, w1, b1)
            h2 = relu(h1)
            h3 = fc(h2, w2, b2)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h3, y)
            #print(l)
            #print(dl_dy)
            ll[i] = l

            # backward propagation
            dl_dh2, dl_dw2, dl_db2 = fc_backward(dl_dy, h2, w2, b2)
            dl_dh1 = relu_backward(dl_dh2, h1)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dh1, x, w1, b1)

            # accumulate gradients
            dl_dw1_batch += dl_dw1
            dl_db1_batch += dl_db1
            dl_dw2_batch += dl_dw2
            dl_db2_batch += dl_db2

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # accumulate gradients
        w1 -= learning_rate * dl_dw1_batch / batch_size
        b1 -= learning_rate * dl_db1_batch / batch_size
        w2 -= learning_rate * dl_dw2_batch / batch_size
        b2 -= learning_rate * dl_db2_batch / batch_size

    return w1, b1, w2, b2, losses


def train_cnn(mini_batch_x, mini_batch_y, learning_rate=0.05, decay_rate=0.95, num_iters=10000):
    # initialization network weights
    w_conv = 0.1 * np.random.randn(3, 3, 1, 3)
    b_conv = np.zeros((3, 1))
    w_fc = 0.1 * np.random.randn(10, 147)
    b_fc = np.zeros((10, 1))

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training CNN'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
            # print('iter {}/{}'.format(iter + 1, num_iters))

        dl_dw_conv_batch = np.zeros(w_conv.shape)
        dl_db_conv_batch = np.zeros(b_conv.shape)
        dl_dw_fc_batch = np.zeros(w_fc.shape)
        dl_db_fc_batch = np.zeros(b_fc.shape)
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros((batch_size, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]].reshape((14, 14, 1))
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
            h2 = relu(h1)  # (14, 14, 3)
            h3 = pool2x2(h2)  # (7, 7, 3)
            h4 = flattening(h3)  # (147, 1)
            h5 = fc(h4, w_fc, b_fc)  # (10, 1)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h5, y)
            ll[i] = l

            # backward propagation
            dl_dh4, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, h4, w_fc, b_fc)  # (147, 1), (10, 147), (10, 1)
            dl_dh3 = flattening_backward(dl_dh4, h3)  # (7, 7, 3)
            dl_dh2 = pool2x2_backward(dl_dh3, h2)  # (14, 14, 3)
            dl_dh1 = relu_backward(dl_dh2, h1)  # (14, 14, 3)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dh1, x, w_conv, b_conv)  # (3, 3, 1, 3), (3, 1)

            # accumulate gradients
            dl_dw_conv_batch += dl_dw_conv
            dl_db_conv_batch += dl_db_conv
            dl_dw_fc_batch += dl_dw_fc
            dl_db_fc_batch += dl_db_fc

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # update network weights
        w_conv -= learning_rate * dl_dw_conv_batch / batch_size
        b_conv -= learning_rate * dl_db_conv_batch / batch_size
        w_fc -= learning_rate * dl_dw_fc_batch / batch_size
        b_fc -= learning_rate * dl_db_fc_batch / batch_size

    return w_conv, b_conv, w_fc, b_fc, losses


def visualize_training_progress(losses, num_batches):
    # losses - (n_iter, 1)
    num_iters = losses.shape[0]
    num_epochs = math.ceil(num_iters / num_batches)
    losses_epoch = np.zeros((num_epochs, 1))
    losses_epoch[:num_epochs-1, 0] = np.mean(
        np.reshape(losses[:(num_epochs - 1)*num_batches], (num_epochs - 1, num_batches)), axis=1)
    losses_epoch[num_epochs-1] = np.mean(losses[(num_epochs - 1)*num_batches:])

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(range(num_iters), losses), axs[0].set_title('Training loss w.r.t. iteration')
    axs[0].set_xlabel('Iteration'), axs[0].set_ylabel('Loss'), axs[0].set_ylim([0, 5])
    axs[1].plot(range(num_epochs), losses_epoch), axs[1].set_title('Training loss w.r.t. epoch')
    axs[1].set_xlabel('Epoch'), axs[1].set_ylabel('Loss'), axs[1].set_ylim([0, 5])
    fig.suptitle('MLP Training Loss', fontsize=16)
    plt.show()


def infer_mlp(x, w1, b1, w2, b2):
    # x - (m, 1)
    h1 = fc(x, w1, b1)
    h2 = relu(h1)
    h3 = fc(h2, w2, b2)
    y = np.argmax(h3)
    return y


def infer_cnn(x, w_conv, b_conv, w_fc, b_fc):
    # x - (H(14), W(14), C_in(1))
    h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
    h2 = relu(h1)  # (14, 14, 3)
    h3 = pool2x2(h2)  # (7, 7, 3)
    h4 = flattening(h3)  # (147, 1)
    h5 = fc(h4, w_fc, b_fc)  # (10, 1)
    y = np.argmax(h5)
    return y


def compute_confusion_matrix_and_accuracy(pred, label, n_classes):
    # pred, label - (n, 1)
    accuracy = np.sum(pred == label) / len(label)
    confusion = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        for i in range(n_classes):
            # ground true is j but predicted to be i
            confusion[i, j] = np.sum(np.logical_and(label == j, pred == i)) / label.shape[0]
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("Accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


def get_MNIST_data(resource_dir):
    with open(resource_dir + 'mnist_train.npz', 'rb') as f:
        d = np.load(f)
        image_train, label_train = d['img'], d['label']  # (12k, 14, 14), (12k, 1)
    with open(resource_dir + 'mnist_test.npz', 'rb') as f:
        d = np.load(f)
        image_test, label_test = d['img'], d['label']  # (2k, 14, 14), (2k, 1)

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return image_train, label_train, image_test, label_test, label_classes


if __name__ == '__main__':
    work_dir = '/Users/ashwinwariar/CSCI5561/hw5/'

    image_train, label_train, image_test, label_test, label_classes = get_MNIST_data(work_dir)
    image_train, image_test = image_train.reshape((-1, 196)).T / 255.0, image_test.reshape((-1, 196)).T / 255.0

    # Part 1: Multi-layer Perceptron
    # train
    mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    w1, b1, w2, b2, losses = train_mlp(mini_batch_x, mini_batch_y,
                                       learning_rate=0.1, decay_rate=0.9, num_iters=10000)
    visualize_training_progress(losses, len(mini_batch_x))
    np.savez(work_dir + 'mlp.npz', w1=w1, b1=b1, w2=w2, b2=b2)

    # test
    pred_test = np.zeros_like(label_test)
    for i in range(image_test.shape[1]):
        pred_test[i, 0] = infer_mlp(image_test[:, i].reshape((-1, 1)), w1, b1, w2, b2)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    # Part 2: Convolutional Neural Network
    # train
    np.random.seed(0)
    mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    w_conv, b_conv, w_fc, b_fc, losses = train_cnn(mini_batch_x, mini_batch_y,
                                                   learning_rate=0.1, decay_rate=0.9, num_iters=1000)
    visualize_training_progress(losses, len(mini_batch_x))
    np.savez(work_dir + 'cnn.npz', w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)

    # test
    pred_test = np.zeros_like(label_test)
    for i in range(image_test.shape[1]):
        pred_test[i, 0] = infer_cnn(image_test[:, i].reshape((14, 14, 1)), w_conv, b_conv, w_fc, b_fc)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)


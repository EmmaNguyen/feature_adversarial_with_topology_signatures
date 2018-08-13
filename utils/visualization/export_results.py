import matplotlib.pyplot as plt

def save_encoded_sample(data, targets, epoch, vis_path="../data/", figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], -data[:, 1], c=(10 * targets), cmap=plt.cm.Spectral)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    # plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
    plt.savefig("{}/test_latent_epoch_{}.png".format(vis_path, epoch + 1))
    plt.close()
    # save sample input and reconstruction
    vutils.save_image(x,
                      "{}/test_samples_epoch_{}.png".format(vis_path,
                      epoch + 1))
    vutils.save_image(batch['decode'].detach(),
                      "{}/test_reconstructions_epoch_{}.png".format(vis_path, epoch + 1),
                      normalize=True)

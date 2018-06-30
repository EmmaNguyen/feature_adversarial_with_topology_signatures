# This file include a list of customized adversarial learned inference

def run_generative_adversarial_network(opt):
    data_loader = get_MNIST(opt)

    generator = Generator(opt.latent_dim, opt.img_size, opt.channels)
    generator.apply(weights_init_normal)
    generator_solver = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    decoder = Decoder(opt.img_size, opt.latent_dim)
    decoder.apply(weights_init_normal)
    decoder_solver = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    discriminator = Discriminator(opt.img_size, opt.latent_dim)
    discriminator.apply(weights_init_normal)
    discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    adversarial_loss = nn.BCELoss()

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(data_loader):

            if cuda: imgs = imgs.type(torch.cuda.FloatTensor)

            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            real_imgs = Variable(imgs)

            # -----------------
            #  Train Generator
            # -----------------

            generator_solver.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)
            fake_z = decoder(real_imgs)

            # Loss measures generator's ability to fool the discriminator
            # import pdb; pdb.set_trace()
            generator_loss = get_loss_ll
            generator(discriminator, fake_imgs, z, real_imgs, fake_z)
            generator_loss.backward()
            generator_solver.step()

            # ---------------------
            #  Train Discriminatior
            # ---------------------

            discriminator_solver.zero_grad()
            discriminator_loss = get_loss_discriminator(discriminator, fake_imgs, z, real_imgs, fake_z)
            discriminator_loss.backward()
            discriminator_solver.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(data_loader),
                                                            discriminator_loss.data.item(), generator_loss.data.item()))

            batches_done = epoch * len(data_loader) + i

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

if __name__ == "__main__":
    run_generative_adversarial_network(opt)

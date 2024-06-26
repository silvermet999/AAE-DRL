def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)



class Cond(object):
    def __init__(self, data, interval):
        self.model = []
        max_interval = 0
        counter = 0
        self.n_col = 0
        self.n_opt = 0
        self.p = np.zeros((counter, max_interval))
        self.interval = np.asarray(interval)

    def sample_rand(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for i in output_info:
        ed = st + i[0]
        ed_c = st_c + i[0]
        tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none'
        )
        loss.append(tmp)
        st = ed
        st_c = ed_c
    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)

        st = 0
        for item in output_info:
            ed = st + item[0]
            tmp = []
            for j in range(item[0]):
                tmp.append(np.nonzero(data[:, st + j])[0])
            self.model.append(tmp)
            st = ed
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def calc_gradient_penalty(netD, real_data, fake_data, device='gpu', pac=10, lambda_=10):
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


class Synthesizer(SyntheticData):
    def __init__(self,
                 dataset_name,
                 args=None):

        self.dataset_name = dataset_name
        self.in_dim = args.df_sel
        self.hl_dim = args.hl_dim
        self.lr = args.lr

        self.l2scale = args.l2scale
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scores_max = 0

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = dim_reduction.x_pca_train
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)
        out_dim = self.transformer.output_dim
        self.cond_enc_gen = Cond(train_data, self.transformer.output_info)

        self.enc_gen = EncoderGenerator(
            self.in_dim + self.cond_enc_gen.n_opt,
            self.hl_dim,
            out_dim)

        self.dec = Decoder(
            out_dim + self.cond_enc_gen.n_opt,
            self.hl_dim,
            self.in_dim)

        discriminator = Discriminator(
            out_dim + self.cond_enc_gen.n_opt,
            self.hl_dim)

        optimizerEG = Adam(
            self.enc_gen.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=self.l2scale)
        # dec
        optimizerD = Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        if len(train_data) <= self.batch_size:
            self.batch_size = (len(train_data) // 10) * 10

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.in_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size

        for i in range(self.epochs):
            print(i)
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_enc_gen.sample_rand(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1)
                    m1 = torch.from_numpy(m1)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.enc_gen(fakez)

                real = torch.from_numpy(real.astype('float32'))

                if c1 is not None:
                    fake_cat = torch.cat([fake, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                loss_d = -torch.mean(y_real) + torch.mean(y_fake)
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat)

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_enc_gen.sample_rand(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1)
                    m1 = torch.from_numpy(m1)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.enc_gen(fakez)

                recon_loss = F.mse_loss(fake, fakez)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fake, c1], dim=1))
                else:
                    y_fake = discriminator(fake)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_eg = -torch.mean(y_fake) + recon_loss + cross_entropy

                optimizerEG.zero_grad()
                loss_eg.backward()
                optimizerEG.step()

    def sample(self, n):

        self.enc_gen.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.in_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            condvec = self.cond_enc_gen.sample_rand(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.enc_gen(fakez)
            data.append(fake.detach().gpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from PIL import Image
from torchvision import transforms
import os

from src.data_processing import data_loading
from src.pipeline import logger as training_logger
from src.model import architecture
from src.model import losses


class Pipeline:
    def __init__(
            self, dataloader, model, gen_criterion, disc_criterion,
            gen_optimizer, disc_optimizer, logger, config
    ):
        self.dataloader = dataloader
        self.model = model
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.logger = logger
        self.config = config
        self.counter = 0

    def train_model(self):
        for epoch in range(self.config.epochs):
            self.counter = 0
            self.run_epoch(epoch)

    def save_model(self, epoch):
        if (epoch % self.config.save_model_interval == 0) and epoch:
            save_folder = Path(self.config.save_model_path.format(
                    ds_name=self.config.ds_name,
                    model_architecture=self.config.model_architecture,
                    hparams=self.config.hparams_str,
            ))
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder / f"checkpoint_{epoch}.pth")
            torch.save(self.model.state_dict(), save_path)

    def save_img(self, epoch, real_img, img_gen, latent=None, y=None):
        if epoch % self.config.save_metric_interval == 0 and self.counter == 0:
            with torch.no_grad():
                fake = img_gen.detach().cpu()[:self.config.save_img_count, ...]
            fake_img = np.transpose(vutils.make_grid(
                fake, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
            plt.imshow(fake_img)

            file_name = f"ep{epoch}_step{self.counter}.png"
            gen_imgs_save_folder = Path(self.config.gen_imgs_save_path.format(
                ds_name=self.config.ds_name,
                model_architecture=self.config.model_architecture,
                hparams=self.config.hparams_str,
            ))
            gen_imgs_save_folder.mkdir(parents=True, exist_ok=True)
            gen_imgs_save_path = str(gen_imgs_save_folder / file_name)
            plt.savefig(fname=gen_imgs_save_path)

            if latent is not None:
                img_gen, noise = self.model.generate_imgs(cls=y, noise=latent)
                img_gen = img_gen.detach().cpu()[:self.config.save_img_count, ...]
                img_gen = np.transpose(vutils.make_grid(
                    img_gen, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
                plt.imshow(img_gen)

                file_name = f"ep{epoch}_step{self.counter}_reconstructed.png"
                gen_imgs_save_folder = Path(self.config.gen_imgs_save_path.format(
                    ds_name=self.config.ds_name,
                    model_architecture=self.config.model_architecture,
                    hparams=self.config.hparams_str,
                ))
                gen_imgs_save_folder.mkdir(parents=True, exist_ok=True)
                gen_imgs_save_path = str(gen_imgs_save_folder / file_name)
                plt.savefig(fname=gen_imgs_save_path)

        self.counter += 1


class BigBiGANPipeline(Pipeline):
    def run_epoch(self, epoch):
        for step, (x, y) in tqdm(enumerate(self.dataloader)):
            x, y = x.to(device=self.config.device), y.to(device=self.config.device)
            self.model.req_grad_disc(True)
            for _ in range(self.config.disc_steps):
                img_gen, noise = self.model.generate_imgs(cls=y)
                z_img = self.model.generate_latent(img=x)
                self.disc_optimizer.zero_grad()
                outputs = self.model.forward(
                    img_real=x,
                    img_gen=img_gen.detach(),
                    z_noise=noise,
                    z_img=z_img.detach(),
                    cls=y
                )
                disc_loss = self.disc_criterion(outputs)
                disc_loss.backward()
                self.disc_optimizer.step()

            self.model.req_grad_disc(False)
            self.gen_optimizer.zero_grad()
            outputs = self.model.forward(img_real=x, img_gen=img_gen, z_noise=noise, z_img=z_img, cls=y)
            gen_enc_loss = self.gen_criterion(outputs)
            gen_enc_loss.backward()
            self.gen_optimizer.step()

            self.save_img(epoch, x, img_gen, z_img, y)
            self.save_model(epoch)
            self.logger(epoch, step, disc_loss, gen_enc_loss)

    @classmethod
    def from_config(cls, data_path, config):
        config.device = torch.device(config.device)
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigBiGAN.from_config(config).to(device=config.device)

        gen_enc_criterion = losses.GeneratorEncoderLoss()
        disc_criterion = losses.BiDiscriminatorLoss()

        gen_enc_optimizer = torch.optim.Adam(model.get_gen_enc_params(), lr=config.lr_gen, betas=config.betas)
        disc_optimizer = torch.optim.Adam(model.get_disc_params(), lr=config.lr_disc, betas=config.betas)

        logger = training_logger.BiGANLogger.from_config(config=config, name=config.hparams_str)
        return cls(
            model=model,
            gen_criterion=gen_enc_criterion,
            disc_criterion=disc_criterion,
            gen_optimizer=gen_enc_optimizer,
            disc_optimizer=disc_optimizer,
            dataloader=dataloader,
            logger=logger,
            config=config,
        )


class BigBiGANInference:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    #Original
    def inference(self):
        for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
            org_img = org_img.to(self.config.device)
            y = y.to(self.config.device)
            latent = self.encode(org_img)
            print("Latent shape: ", latent.shape)
            reconstructed_img = self.generate(y, latent)
            self.save_img(org_img, reconstructed_img)
            break

    def inference_fromdt(self, specific_dataloader):
        for step, (org_img, y) in tqdm(enumerate(specific_dataloader)):
            org_img = org_img.to(self.config.device)
            y = y.to(self.config.device)
            latent = self.encode(org_img)
            print("Latent shape: ", latent.shape)
            reconstructed_img = self.generate(y, latent)
            self.save_img(org_img, reconstructed_img)
            break

    def inference_my(self,mydataloader):
        for step, (org_img, y) in tqdm(enumerate(mydataloader)):
            org_img = org_img.to(self.config.device)
            y = y.to(self.config.device)
            latent = self.encode(org_img)
            reconstructed_img = self.generate(y, latent)
            #PLOT ORIGINAL AND RECONSTRUCTED IMAGE
            plt.figure()
            plt.imshow(org_img[0].detach().cpu().numpy().transpose(1, 2, 0))
            plt.title("Original Image")
            plt.show()
            plt.figure()
            img = reconstructed_img[0].detach().cpu()[:self.config.save_img_count, ...]
            img = np.transpose(vutils.make_grid(
                img, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
            plt.imshow(img)
            plt.title("Reconstructed Image")

            #self.save_img(org_img, reconstructed_img)
            break
    
    def inference_save(self, mydataloader, label):
        # Method to save the input img and the output img
        # Input: ./data/FMNIST/input/{timestamp}.png
        # Output: ./data/FMNIST/output/{timestamp}.png
        for step, (org_img, y) in tqdm(enumerate(mydataloader)):
            org_img = org_img.to(self.config.device)
            y = y.to(self.config.device)
            latent = self.encode(org_img)
            reconstructed_img = self.generate(y, latent)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            input_dir = "./data/FMNIST/" +  label + "/input/"
            output_dir = "./data/FMNIST/"+ label +"/output/"

            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #save imgs

            input_img = (org_img[0].clamp(-1, 1) + 1) / 2.0
            input_pil = transforms.ToPILImage()(input_img[0].detach().cpu())
            input_pil = input_pil.convert("L")
            input_pil.save(input_dir + timestamp + ".png")

            output_img = (reconstructed_img[0].clamp(-1, 1) + 1) / 2.0
            output_pil = transforms.ToPILImage()(output_img[0].detach().cpu())
            output_pil = output_pil.convert("L")
            output_pil.save(output_dir + timestamp + ".png")




    def encode_batch(self, batch):
        latent_array = []
        for step, (org_img, y) in tqdm(enumerate(batch)):
            latent = self.encode(org_img)
            latent_array.append(latent)
        print("Latent array length: ", len(latent_array)) 


    def encode_sample(self, no_samples):
        #encode no_samples images
        lante_array = []
        for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
            if step == no_samples:
                break
            org_img = org_img.to(self.config.device)
            y = y.to(self.config.device)
            class_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
            string_labels = [class_mapping[label.item()] for label in y]
            print("String labels: ", string_labels)
            #append tuple with latent and label each item with labels
            # Layout ["Latent": tensor, "Label": string]
            print("Img shape: ", org_img.shape)
            print("Img dtype: ", org_img.dtype)
            lante_array.append(self.encode(org_img))
        return lante_array
        

    def encode(self, img):
        z_img = self.model.generate_latent(img=img)
        return z_img

    def generate(self, y, latent):
        img_gen, noise = self.model.generate_imgs(cls=y, noise=latent)
        return img_gen, noise

    def save_img(self, org_img, reconstructed_img):
        for name, img in [("org_img", org_img), ("reconstructed_img", reconstructed_img)]:
            img = img[0].detach().cpu()[:self.config.save_img_count, ...]
            img = np.transpose(vutils.make_grid(
                img, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
            #add label to image
            plt.title(name)
            plt.imshow(img)
            file_name = f"{name}.png"
            gen_imgs_save_folder = Path(self.config.rec_imgs_save_path.format(
                ds_name=self.config.ds_name,
                model_architecture=self.config.model_architecture,
                hparams=self.config.hparams_str,
            ))
            gen_imgs_save_folder.mkdir(parents=True, exist_ok=True)
            gen_imgs_save_path = str(gen_imgs_save_folder / file_name)
            plt.savefig(fname=gen_imgs_save_path)

    @classmethod
    def from_checkpoint(cls, data_path, checkpoint_path, config):
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigBiGAN.from_config(config).to(device=config.device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=True)
        model = model.cuda()
        model = model.eval()
        return cls(model=model, dataloader=dataloader, config=config)


class GANPipeline(Pipeline):
    def run_epoch(self, epoch):
        for step, (x, y) in tqdm(enumerate(self.dataloader)):
            x, y = x.to(device=self.config.device), y.to(device=self.config.device)
            if self.model.cls is None: self.model.cls = y.detach()
            img_gen, noise = self.model.generate_imgs(cls=y)

            self.model.req_grad_disc(True)
            disc_loss, disc_real_acc, disc_fake_acc = self.forward_disc(x, img_gen, y)
            self.model.req_grad_disc(False)
            gen_loss, gen_disc_acc = self.forward_gen(img_gen, y, noise)

            self.save_img(epoch, x, img_gen)
            if (epoch % self.config.save_model_interval == 0) and epoch:
                torch.save(self.model.state_dict(), self.config.save_model_path)
            self.logger(epoch, step, disc_loss, gen_loss, gen_disc_acc, disc_real_acc, disc_fake_acc)

    def forward_gen(self, gen_img, y, noise):
        for i in range(self.config.gen_steps):
            self.model.generator.zero_grad()
            _, pred_gen_img = self.model.discriminator(x=gen_img, cls=y)
            pred_gen_img = torch.sigmoid(pred_gen_img.reshape(-1))

            label_gen_img = torch.ones(pred_gen_img.shape[0], device=self.config.device)
            gen_loss = self.gen_criterion(pred_gen_img, label_gen_img)
            gen_loss.backward()

            gen_disc_acc = 1 - pred_gen_img.mean().item()
            self.gen_optimizer.step()

            if self.config.gen_steps > 1:
                gen_img, _ = self.model.generate_imgs(cls=y, noise=noise)

        return gen_loss, gen_disc_acc

    def forward_disc(self, img, gen_img, y):
        for _ in range(self.config.disc_steps):
            self.model.discriminator.zero_grad()
            _, pred_real_img = self.model.discriminator(x=img, cls=y)
            pred_real_img = torch.sigmoid(pred_real_img.reshape(-1))

            label_real_img = torch.ones(pred_real_img.shape[0], device=self.config.device)
            real_img_loss = self.disc_criterion(pred_real_img, label_real_img)
            real_img_loss.backward()

            _, pred_gen_img = self.model.discriminator(x=gen_img.detach(), cls=y)
            pred_gen_img = torch.sigmoid(pred_gen_img.reshape(-1))

            label_gen_img = torch.zeros(pred_gen_img.shape[0], device=self.config.device)
            gen_img_loss = self.disc_criterion(pred_gen_img, label_gen_img)
            gen_img_loss.backward()

            disc_real_acc = pred_real_img.mean().item()
            disc_fake_acc = 1 - pred_gen_img.mean().item()

            disc_loss = gen_img_loss + real_img_loss

            self.disc_optimizer.step()

        return disc_loss, disc_real_acc, disc_fake_acc

    @classmethod
    def from_config(cls, data_path, config):
        config.device = torch.device(config.device)
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigGAN.from_config(config).to(device=config.device)

        gen_criterion = torch.nn.BCELoss()
        disc_criterion = torch.nn.BCELoss()

        gen_optimizer = torch.optim.Adam(model.get_gen_params(), lr=config.lr_gen, betas=config.betas)
        disc_optimizer = torch.optim.Adam(model.get_disc_params(), lr=config.lr_disc, betas=config.betas)

        logger = training_logger.GANLogger.from_config(config=config, name=config.hparams_str)
        return cls(
            model=model,
            gen_criterion=gen_criterion,
            disc_criterion=disc_criterion,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            dataloader=dataloader,
            logger=logger,
            config=config,
        )


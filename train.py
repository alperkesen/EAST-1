import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

	test_img_path = os.path.abspath('../ICDAR_2015/test_img')
	test_gt_path  = os.path.abspath('../ICDAR_2015/test_gt')

	file_num2 = len(os.listdir(test_img_path))
	testset = custom_dataset(test_img_path, test_gt_path)
	test_loader = data.DataLoader(testset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	try:
		checkpoint = torch.load('./pths/east.pth')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch_dict = checkpoint['epoch_loss']
		test_dict = checkpoint['test_loss']
		total_epoch = checkpoint['epoch']
		best_loss = checkpoint['best_loss']
		best_acc = checkpoint['best_acc']
	except FileNotFoundError:
		model.load_state_dict(torch.load('./pths/east_vgg16.pth'))
		epoch_dict = dict()
		test_dict = dict()
		total_epoch = 0
		best_loss = float('inf')
		best_acc = 0

	print("Continue from epoch {}".format(total_epoch))
	print("Epoch_dict", epoch_dict)
	print("Test_dict", test_dict)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	for epoch in range(epoch_iter):	
		model.train()
		scheduler.step()
		epoch_loss = 0
		test_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

		epoch_dict[total_epoch + epoch + 1] = (epoch_loss / int(file_num / batch_size), epoch_loss)
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}, epoch_loss: {}'.format(epoch_loss / int(file_num/batch_size), time.time()-epoch_time, epoch_loss))


		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

			test_loss += loss.item()
			print('Epoch (test) is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num2/batch_size), time.time()-start_time, loss.item()))

		test_dict[total_epoch + epoch + 1] = (test_loss / int(file_num2 / batch_size), test_loss)
		print('test_loss is {:.8f}, epoch_time is {:.8f}, test_loss: {}'.format(test_loss / int(file_num2/batch_size), time.time()-epoch_time, test_loss))

		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			model_state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save({
				'epoch': total_epoch + epoch + 1,
				'model_state_dict': model_state_dict,
				'optimizer_state_dict': optimizer.state_dict(),
				'epoch_loss': epoch_dict,
				'test_loss': test_dict,
				'best_loss': best_loss,
				'best_acc': best_acc
      }, os.path.join(pths_path, 'east.pth'))

		if (total_epoch + epoch + 1) % 10 == 0:
			model_state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save({
				'epoch': total_epoch + epoch + 1,
				'model_state_dict': model_state_dict,
				'optimizer_state_dict': optimizer.state_dict(),
				'epoch_loss': epoch_dict,
				'test_loss': test_dict,
				'best_loss': best_loss,
				'best_acc': best_acc
      }, os.path.join(pths_path, 'east_epoch_{}.pth'.format(total_epoch + epoch + 1)))

		if test_loss / int(file_num2/batch_size) < best_loss:
			model_state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save({
				'epoch': total_epoch + epoch + 1,
				'model_state_dict': model_state_dict,
				'optimizer_state_dict': optimizer.state_dict(),
				'epoch_loss': epoch_dict,
				'test_loss': test_dict,
				'best_loss': best_loss,
				'best_acc': best_acc
      }, os.path.join(pths_path, 'east_best_loss.pth'))


if __name__ == '__main__':
	train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	pths_path      = './pths'
	batch_size     = 16
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 2
	save_interval  = 1
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)	
	

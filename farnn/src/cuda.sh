#!/bin/bash

if lspci | grep -i nvidia && gcc --version; then
	luarocks install cutorch;
	printf "\n\n Installing CUDA"
	echo "Details here: http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3vTMAvnnO"
	cd ~/Downloads;
	wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
	wait
	#Verify conflicting installation methods
	 sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_*.*;
	wait
	sudo dpkg -i cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
	wait
	sudo apt-get install cuda
	wait

	```bash
		The PATH variable needs to include /usr/local/cuda-7.0/bin
		Setting it up 'for ya'
	```
	if uname -m == x86_64; then
		$(export PATH=/usr/local/cuda-7.0/bin:$PATH)
		$(export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH)
	else  #[[ $(uname -m) -ne x86_64 ]]; then  #assume 32-bit OS
		$(export PATH=/usr/local/cuda-7.0/bin:$PATH)
		$(export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_PATH)
		#statements
	fi
else 

	   printf "\n\nYou have no cuda capable gpu. Exiting the Cuda installation loop.\n\n"
		
fi
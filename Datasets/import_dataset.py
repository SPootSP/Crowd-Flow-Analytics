import requests

def download_shanghaitech_dataset(store_dir):
	# incase the download link does not work:
	# manual download link = https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&e=1&dl=0
	url = "https://uc9873bff7e1214fb3d1ddc120c7.dl.dropboxusercontent.com/cd/0/get/CUh2yPazkXorOrwowJbGFnKQTFyx9XPA7TRekyk8ADNKdCQbId5V8BMUsycQ-zvBNOjN9cvvXB5plvvUJkgz8ebsPl1f3cNC4ztVVAa02cswY5jF-732TWIeUdrvXcYtfxUu546DfQlSPCMtt8M10jvMQBAf3txKyR8s4eQd1zjUZg/file?_download_id=228827871708020339640333567711943143542007834196408654549559916&_notify_domain=www.dropbox.com&dl=1"
	response = requests.get(url)

	with open(store_dir, mode="wb") as file:
		file.write(response.content)

def download_jhucrowdv2_0_dataset(store_dir):
	# incase the download link does not work:
	# manual download link = https://drive.google.com/drive/folders/1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n
	url = "https://drive.usercontent.google.com/download?id=1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj&export=download&confirm=t&uuid=f3aabc30-4975-46dc-b118-2c6ac13b067d"
	response = requests.get(url)

	with open(store_dir, mode="wb") as file:
		file.write(response.content)
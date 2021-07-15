const supported = 'mediaDevices' in navigator;
let imgs = []
let imageCapture
if(!supported) {
	alert("No supported camera found on device")
} else {
	//<video id="player" controls autoplay></video>
	const player = document.getElementById("player")
	//const canvas = document.getElementById('canvas');
	//const context = canvas.getContext('2d');
	const captureButton = document.getElementById('capture');

	let size = window.innerWidth;
	if(size > window.innerHeight)
		size = window.innerHeight - 200
	player.width  = size
	player.height = size
	const constraints = {
		video: {
			facingMode: "environment",
			width: { ideal: 480 },
    		height: { ideal: 480 }
		}
	};

	navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
		player.srcObject = stream;

		const track = stream.getVideoTracks()[0];
    	imageCapture = new ImageCapture(track);
	});
}

function capture() {
	document.getElementById("capture").setAttribute("disabled", true)
	imageCapture.takePhoto()
	  	.then(blob => createImageBitmap(blob))
	  	.then(imageBitmap => {
			if(imgs.length == 0) {
				document.getElementById("capture").removeAttribute("disabled")
				imgs.push(imageBitmap)
				document.getElementById("instruction").innerHTML = "Please take a photo of the opposite corner of the cube"
			}
			else {
				imgs.push(imageBitmap)
				let cube_state = extractCubeStateFromImgs(imgs)
				document.location.href = "solve.html?state=" + cube_state;
			}
	  	})
	  	.catch(error => console.log(error));
}

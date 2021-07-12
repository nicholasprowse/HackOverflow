
function test(){
	const im = new Image();
	im.onload = () => {
		console.log(im)
	    let a = tf.browser.fromPixels(im)
		// Want img to have maximum size of 500
		scale = 500 / Math.max(a.shape[0], a.shape[1])
		a = tf.image.resizeBilinear(a, [a.shape[0] * scale, a.shape[1] * scale])
		a = a.div(255);
		edges = cannyEdgeDetector(a);
		showImg(edges);
	}
	im.src = "../imgs/cube_1.jpeg";
}

function showImg(img) {
	canvas = document.createElement('canvas');
	document.body.insertBefore(canvas, null);
	tf.browser.toPixels(img, canvas);
}

// img has shape [W, H, 3]
function cannyEdgeDetector(img) {
	// First, blur image with 5x5 Gaussian kernel to remove noise
	const blur = tf.tensor([[2,  4,  5,  4, 2],
     					        [4,  9, 12,  9, 4],
     	   				        [5, 12, 15, 12, 5],
         				        [2,  4,  5,  4, 2],
     	   				        [4,  9, 12,  9, 4]]).div(159);
	const zero = tf.zeros([5, 5]);
	const kernel = tf.stack([tf.stack([blur, zero, zero], 2),
						     tf.stack([zero, blur, zero], 2),
						     tf.stack([zero, zero, blur], 2)], 3)
	let blurred = tf.conv2d(img, kernel, strides=[1, 1], pad='valid');
	return blurred;
}

// Possible improvements:
// Use hysteresis to include weak edges that are attached to strong edges
// Replace Gaussian filter with an adaptive filter to prevent smoothing out edgesToImage

// img has shape [W, H, 3]
function cannyEdgeDetector(img) {
	// First, blur image with 5x5 Gaussian kernel to remove noise
	const blurKernel = tf.tensor([
		[2, 4, 5, 4, 2],
		[4, 9, 12, 9, 4],
		[5, 12, 15, 12, 5],
		[2, 4, 5, 4, 2],
		[4, 9, 12, 9, 4]
	]).div(159);
	let blurred = convolve(img, blurKernel)

	// Calculate gradient along each direction
	const xKernel = tf.tensor([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]
	]);
	const yKernel = tf.tensor([
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]
	]);
	let xGrad = convolve(blurred, xKernel)
	let yGrad = convolve(blurred, yKernel)

	// Compute the magnitude and direction of the gradient
	let grad = tf.add(xGrad.square(), yGrad.square()).arraySync()
	let dir = tf.atan2(yGrad, xGrad).mul(180/Math.PI)
	dir = dir.div(45).round().mul(45).arraySync()

	// For each pixel, determine if it is an edge.
	// A pixel is an edge if its gradient is above the threshold, and both
	// neighboring pixels along the gradient direction (perpendicular to edge)
	// have a smaller gradient direction
	let edges = [];
	const threshold = 0.15;
	const h = xGrad.shape[0], w = xGrad.shape[1];
	for(c = 0; c < 3; c++) {
		for(y = 0; y < h; y++) {
		    for(x = 0; x < w; x++) {
		        if(grad[y][x][c] < threshold)
		            continue;

		        angle = dir[y][x][c] * Math.PI/180;
		        const dx = Math.round(Math.cos(angle));
		        const dy = Math.round(Math.sin(angle));

		        if(0 <= y+dy && y+dy < h && 0 <= y-dy && y-dy < h)
		            if(0 <= x+dx && x+dx < w && 0 <= x-dx && x-dx < w)
		                if(grad[y+dy][x+dx][c] < grad[y][x][c] && grad[y-dy][x-dx][c] < grad[y][x][c])
		                    // We lose 3 pixels on each edge when performing the convolutions, so we add those back in here
							edges.push([x+3, y+3]);
		    }
		}
	}

	return edgesToImage(img, edges);
}

// Scales all elements in a tensor to be in the range 0 to 1 for displaying as image
function normalise(img) {
	img = img.sub(img.min())
	return img.div(img.max())
}

// Tensorflow kernel requires input and output channel dimensions, but ijust
// want to apply the same kernel to all channels independantly, so this function
// generates the appropriate kernel to achieve that.
function convolve(img, kernel) {
	const zero = tf.zerosLike(kernel)
	kernel = tf.stack([
		tf.stack([kernel, zero, zero], 2),
		tf.stack([zero, kernel, zero], 2),
		tf.stack([zero, zero, kernel], 2)
	], 3)
	return tf.conv2d(img, kernel, strides = [1, 1], pad = 'valid')
}

//************ THESE FUNCTIONS ARE ALL FOR TESTING PURPOSES ONLY ***************

function edgesToImage(orig, edges) {
	orig = orig.bufferSync();
	for(i = 0; i < edges.length; i++) {
		orig.set(0, edges[i][1], edges[i][0], 0)
		orig.set(0, edges[i][1], edges[i][0], 1)
		orig.set(0, edges[i][1], edges[i][0], 2)
	}
	return orig.toTensor()
}

function test() {
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
	img = normalise(img);
	canvas = document.createElement('canvas');
	document.body.insertBefore(canvas, null);
	tf.browser.toPixels(img, canvas);
}

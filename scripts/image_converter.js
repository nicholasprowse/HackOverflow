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
	const h = xGrad.shape[0], w = xGrad.shape[1];
	let mask = tf.buffer([h, w], 'bool')	// To keep track of which edges have been added
	let edges = [];
	const threshold = 0.15;
	for(let c = 0; c < 3; c++) {
		for(let y = 0; y < h; y++) {
		    for(let x = 0; x < w; x++) {
		        if(grad[y][x][c] < threshold || mask.get(y, x))
		            continue;

		        angle = dir[y][x][c] * Math.PI/180;
		        const dx = Math.round(Math.cos(angle));
		        const dy = Math.round(Math.sin(angle));

		        if(0 <= y+dy && y+dy < h && 0 <= y-dy && y-dy < h)
		            if(0 <= x+dx && x+dx < w && 0 <= x-dx && x-dx < w)
		                if(grad[y+dy][x+dx][c] < grad[y][x][c] && grad[y-dy][x-dx][c] < grad[y][x][c]) {
		                    // We lose 3 pixels on each edge when performing the convolutions, so we add those back in here
							edges.push([x+3, y+3]);
							mask.set(true, y, x);
						}
		    }
		}
	}

	return edges;
}

/*
 Better idea:
 Store the unrounded gradient direction with each edgel.
 Pick a random edgel. Compute line perpendicular to gradient passing though
 the edgel. Compute the consensus in the same way. If it is below the threshold
 ignore it and try a different random edgel.
 Otherwise refine the line as follows:
 	Take all edgels that agree with the line
	Compute the stdev of the direction of edgels
	Drop any outside of 2 stdev (or some other amount determined empirically)
	Update the line with the mean of the directions of the remaining edgels
	Not sure what the stopping criterion is, since it may be possible that this
	does not converge. Perhaps stop if the line doesn't change or after 5
	updates, which ever comes first?
*/

// Finds n straight line segments from a list of edge pixels
function RANSAC(edges, n) {
	// This algorithm has 3 hyperparameters:
	// d is the maximum distance to a line before an edgel is counted
	// threshReduction is the factor by which the threshold reduces if no lines can be found
	// countThreshold is the number of times to try before reducing the countThreshold
	const d = 2, threshReduction = 0.98, countThreshold = 5000
	lines = []
	let threshold = edges.length / 20
	let count = 0
	while(lines.length < n) {
		// This way of selecting edges ensures that the two edges are not the same
		// but each edge still has the same probability of being selected
		let edge1 = Math.floor(Math.random() * edges.length)
		let edge2 = Math.floor(Math.random() * (edges.length - 1))
		if(edge1 == edge2)
			edge2 = edges.length - 1;

		// Extact edge points, and define the line
		const [u1, v1] = edges[edge1],
			  [u2, v2] = edges[edge2]
		const line = [v1 - v2, u2 - u1, u1*v2 - u2*v1]
		const lineMag = Math.sqrt(line[0]*line[0] + line[1]*line[1])

		// Find all edgels within a distance d of the line
		closeEdges = []
		for(let i = 0; i < edges.length; i++)
			if(Math.abs(edges[i][0]*line[0] + edges[i][1]*line[1] + line[2]) <= d*lineMag)
				closeEdges.push(i)

		// Found a line
		if(closeEdges.length >= threshold) {
			// Add the line, and remove all close edgels so we don't find it again
			lines.push(line);
			for(let i = closeEdges.length - 1; i >= 0; i--)
				edges.splice(closeEdges[i], 1)
			count = 0
		}
		count++

		// If it takes too long, reduce the threshold by threshReduction
		if(count >= countThreshold) {
			threshold = threshReduction * threshold
		}
	}

	return lines;
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

function overlayEdges(img, edges) {
	img = img.bufferSync();
	for(let i = 0; i < edges.length; i++) {
		img.set(1, edges[i][1], edges[i][0], 0)
		img.set(0, edges[i][1], edges[i][0], 1)
		img.set(0, edges[i][1], edges[i][0], 2)
	}
	return img.toTensor()
}

// Warning: Divide by zero errors are not handled
function overlayLines(img, lines) {
	img = img.bufferSync();
	for(const line of lines) {
		const m = -line[0] / line[1];
		if(Math.abs(m) > 1) {
			// Iterate over y
			for(let y = 0; y < img.shape[0]; y++) {
				const x = -Math.round((line[1]*y + line[2]) / line[0])
				if(x >= 0 && x < img.shape[1]) {
					img.set(0, y, x, 0)
					img.set(1, y, x, 1)
					img.set(0, y, x, 2)
				}
			}
		} else {
			// Iterate over x
			for(let x = 0; x < img.shape[1]; x++) {
				const y = -Math.round((line[0]*x + line[2]) / line[1])
				if(y >= 0 && y < img.shape[0]) {
					img.set(0, y, x, 0)
					img.set(1, y, x, 1)
					img.set(0, y, x, 2)
				}
			}
		}
	}
	return img.toTensor();
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
		let edges = cannyEdgeDetector(a);
		let lines = RANSAC(edges, 21);
		// img = overlayEdges(a, edges)
		img = overlayLines(a, lines)
		showImg(img);
	}
	im.src = "../imgs/cube_2.jpeg";
}

function showImg(img) {
	img = normalise(img);
	let canvas = document.createElement('canvas');
	document.body.insertBefore(canvas, null);
	tf.browser.toPixels(img, canvas);
}

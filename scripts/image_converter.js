
function extractCubeStateFromImgs(imgs) {
	// This is incomplete, so just return hardcoded state for testing purposes
	// return "RYRWBOBGGBROWYYRGWGOYWORGWBYRYRYBGWOYWOGBRWOWBOYGGROBB"
	return "GWBYBROGWYBGWORBYOWROWOBOWGWRGRYOBBGWYYGRBOGRRORWGBYYY"
	for(let img of imgs) {
		let edges = cannyEdgeDetector(img)
		let lines = RANSAC(edges, 21)
		let groups = splitLines(lines)
		if(groups == null) {
			console.log("FAILURE")
			return
		}
		let quads = getQuads(img, groups)
	}
}

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
	let dir = tf.atan2(yGrad, xGrad)
	let rDir = dir.mul(4/Math.PI).round().mul(Math.PI/4).arraySync()
	dir = dir.arraySync()

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

		        angle = rDir[y][x][c];
		        const dx = Math.round(Math.cos(angle));
		        const dy = Math.round(Math.sin(angle));

		        if(0 <= y+dy && y+dy < h && 0 <= y-dy && y-dy < h)
		            if(0 <= x+dx && x+dx < w && 0 <= x-dx && x-dx < w)
		                if(grad[y+dy][x+dx][c] < grad[y][x][c] && grad[y-dy][x-dx][c] < grad[y][x][c]) {
		                    // We lose 3 pixels on each edge when performing the convolutions, so we add those back in here
							edges.push([x+3, y+3, dir[y][x][c]]);
							mask.set(true, y, x);
						}
		    }
		}
	}

	return edges;
}

function brightenColor(color) {
	let [r, g, b] = color
	let max = Math.max(r, g, b), min = Math.min(r, g, b)
	let delta = max - min
	let h = 0
	if(max == r)
		h = ((g - b) / delta) % 6
	if(max = g)
		h = ((b - r) / delta) + 2
	if(max = b)
		h = ((r - g) / delta) + 4
	h *= 60
	let s = 0
	if(max > 0)
		s = delta / max


	let x = s * (1 - Math.abs((h/60) % 2 - 1))
	if(h < 60)
		[r, g, b] = [s, x, 0]
	else if(h < 120)
		[r, g, b] = [x, s, 0]
	else if(h < 180)
		[r, g, b] = [0, s, x]
	else if(h < 240)
		[r, g, b] = [0, x, s]
	else if(h < 300)
		[r, g, b] = [x, 0, s]
	else
		[r, g, b] = [s, 0, x]
	return [r, g, b]
}

// Finds n straight line segments from a list of edge pixels
function RANSAC(edges, n) {
	// This algorithm has 3 hyperparameters:
	// d is the maximum distance to a line before an edgel is counted
	// threshReduction is the factor by which the threshold reduces if no lines can be found
	// countThreshold is the number of times to try before reducing the countThreshold
	const d = 2, threshReduction = 0.98, countThreshold = 1000
	lines = []
	let threshold = edges.length / 15
	let count = 0
	while(lines.length < n) {
		// Select random edgel and define line passing through edgel
		// perpedicular to the edgels gradient
		const [x, y, a] = edges[Math.floor(Math.random() * edges.length)]
		const m = -1/Math.tan(a)		// Gradient of line
		const line = [m/(y-m*x), -1/(y-m*x), 1]
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
		if(count >= countThreshold)
			threshold = threshReduction * threshold

	}
	return lines;
}

function kmeans(data, k, means=null) {
	let n = data.length
	for(let i = 0; i < n; i++)
		data[i] = tf.tensor(data[i])

	if(means == null) {
		means = new Array(k)
		for(let i = 0; i < k; i++)
			means[i] = data[Math.floor(Math.random() * data.length)]
	}

	while(true) {
	    // Create label array containing which mean each pixel is closest to
	    let label = new Array(n);
	    for(let i = 0; i < n; i++) {
			let min = 0, minDist = 1e10
			for(let j = 0; j < k; j++) {
				const dist = data[i].squaredDifference(means[j]).sum().dataSync()[0]
		        if(dist < minDist){
					minDist = dist
					min = j
				}
			}
			label[i] = min
	    }

	    // Recompute means using the new labels
	    new_means = new Array(k)
	    total = new Array(k)
		for(let i = 0; i < k; i++){
			new_means[i] = tf.zerosLike(data[0])
			total[i] = 0
		}
	    for(let i = 0; i < n; i++) {
	        new_means[label[i]] = new_means[label[i]].add(data[i])
	        total[label[i]]++
	    }

		let finished = true
		for(let i = 0; i < k; i++) {
			new_means[i] = new_means[i].div(total[i])
			if(new_means[i].notEqual(means[i]).sum().dataSync()[0])
				finished = false
			means[i] = new_means[i]
		}

		if(finished) {
			for(let i = 0; i < n; i++)
				data[i] = data[i].dataSync()
			return label
		}
	}
}

function getQuads(img, groups) {
	// Sort each group so the lines are in order
	for(let i = 0; i < 3; i++)
		groups[i] = sortGroup(groups[i], img.shape[1]/2, img.shape[0]/2)

	let [quads, x1] = getQuadsFromFace(groups, 0, false)
	let [quads2, x2] = getQuadsFromFace(groups, 0, true)

	quads = quads.concat(quads2)
	let x = []
	for(let i = 2; i < 6; i++)
		if(i != x1 && i != x2)
			x.push(i)

	let g1 = Math.floor(x[0] / 2),
		g2 = Math.floor(x[1] / 2)

	let off1 = (x[0] % 2) == 0 ? 0 : 3
	let off2 = (x[1] % 2) == 0 ? 0 : 3
	for(let i = 0; i < 3; i++) {
		for(let j = 0; j < 3; j++) {
			quads.push([
				groups[g1][i+off1], groups[g2][j+off2],
				groups[g1][i+off1+1], groups[g2][j+off2+1]])
		}
	}
	return quads
}

/*
 Given 3 groups of lines, this will return the quads on the face attached to the
 given line. group is the index of the group that line belongs to, and start
 determines if the line of interest is at the start or end of the group
*/
function getQuadsFromFace(groups, g, start) {
	let others = [0, 1, 2]
	others.splice(g, 1)
	let l = groups[g][start ? 0 : 6]
	let ref = [(-10000*l[1] - l[2])/l[0], 10000]
	let d = new Array(4), p = new Array(4)
	for(let i = 0; i < 4; i++) {
		p[i] = intersection(l, groups[others[Math.floor(i/2)]][6*(i%2)])
		d[i] = dist(ref, p[i])
	}
	let index = [...Array(4).keys()]
	index.sort((a, b) => d[a] - d[b])
	let a = p[index[1]], b = p[index[2]]

	let groupValues = new Array(6).fill(1e10)
	for(let i = 0; i < 3; i++) {
		if(i == g)
			continue

		let ints = [intersection(l, groups[i][0]),
					intersection(l, groups[i][3]),
					intersection(l, groups[i][6])]

		d = new Array(3)
		for(let j = 0; j < 3; j++)
			d[j] = Math.min(dist(ints[j], a), dist(ints[j], b))

		groupValues[2*i] = d[0] + d[1]
		groupValues[2*i+1] = d[1] + d[2]
	}

	let x = tf.tensor(groupValues).argMin().arraySync()
	let g2 = Math.floor(x/2)
	let off1 = start ? 0 : 3
	let off2 = (x % 2) == 0 ? 0 : 3

	let quads = []
	for(let i = 0; i < 3; i++)
		for(let j = 0; j < 3; j++)
			quads.push([groups[g][i+off1], groups[g2][j+off2],
					    groups[g][i+off1+1], groups[g2][j+off2+1]])

	return [quads, x]
}

// Determines if p is inside quad defined by 4 lines
// l1 and l3 should be on opposite sides of quad
function inQuad(p, l1, l2, l3, l4) {
	let p1 = cross(l1, l2),
		p2 = cross(l2, l3),
		p3 = cross(l3, l4),
		p4 = cross(l4, l1)

	p1 = [p1[0]/p1[2], p1[1]/p1[2]]
	p2 = [p2[0]/p2[2], p2[1]/p2[2]]
	p3 = [p3[0]/p3[2], p3[1]/p3[2]]
	p4 = [p4[0]/p4[2], p4[1]/p4[2]]

	return inTriangle(p, p1, p2, p3) || inTriangle(p, p2, p3, p4)
}

// Determines if p is inside triangle defined by 3 corner points
function inTriangle(p, a, b, c) {
	// Compute vectors
	let v0 = [c[0] - a[0], c[1] - a[1]],
		v1 = [b[0] - a[0], b[1] - a[1]],
		v2 = [p[0] - a[0], p[1] - a[1]]

	// Compute dot products
	let dot00 = dot(v0, v0),
		dot01 = dot(v0, v1),
		dot02 = dot(v0, v2),
		dot11 = dot(v1, v1),
		dot12 = dot(v1, v2)

	// Compute barycentric coordinates
	let invDenom = 1 / (dot00 * dot11 - dot01 * dot01),
		u = (dot11 * dot02 - dot01 * dot12) * invDenom,
		v = (dot00 * dot12 - dot01 * dot02) * invDenom

	// Check if point is in triangle
	return (u > 0) && (v > 0) && (u + v < 1)
}

function dot(a, b) {
	let sum = 0;
	for(let i = 0; i < a.length; i++)
		sum += a[i] * b[i];
	return sum;
}

// Scales all elements in a tensor to be in the range 0 to 1 for displaying as image
function normalise(img) {
	img = img.sub(img.min())
	return img.div(img.max())
}

// Tensorflow kernel requires input and output channel dimensions, but I just
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

function splitLines(lines) {
	let groups = [[], [], []]
	let count = 0
	do {
		let angles = new Array(lines.length)
		for(let i = 0; i < angles.length; i++) {
			let a = Math.atan2(-lines[i][0], lines[i][1])
			angles[i] = [Math.cos(a), Math.sin(a)]
		}
		let labels = kmeans(angles, 3)
		groups = [[], [], []]
		for(let i = 0; i < lines.length; i++)
			groups[labels[i]].push(lines[i])
		if(count > 50)
			return null
		count++
	} while(groups[0].length != 7 || groups[1].length != 7 || groups[2].length != 7)
	return groups
}

function sortGroup(lines, x, y) {
	let [a, b, c] = lines[0]
	let perp = [-(b/(b*x - a*y)), -(a/(-b*x + a*y)), 1]
	let ref = [(-10000*perp[1] - perp[2])/perp[0], 10000]
	let d = new Array(lines.length)
	for(let i = 0; i < lines.length; i++)
		d[i] = dist(ref, intersection(perp, lines[i]))

	let index = [...Array(lines.length).keys()]
	index.sort((a, b) => d[a] - d[b])
	let newLines = new Array(lines.length)
	for(let i = 0; i < lines.length; i++)
		newLines[i] = lines[index[i]]

	return newLines
}

function dist(a, b){
	let sum = 0
	for(let i = 0; i < a.length; i++)
		sum += (a[i] - b[i])**2
	return sum
}

function intersection(a, b) {
	let [x, y, l] = cross(a, b)
	return [x/l, y/l]
}

function cross(a, b) {
	let [x, y, z] = a, [u, v, w] = b
	return [w*y - v*z, -w*x + u*z, v*x - u*y]
}

//************ THESE FUNCTIONS ARE ALL FOR TESTING PURPOSES ONLY ***************

let cube;
let imgSize;
let lines;

function test(image_num) {
	const im = new Image();
	im.onload = () => {
		let img = tf.browser.fromPixels(im)
		// Want img to have maximum size of 500
		scale = 500 / Math.max(img.shape[0], img.shape[1])
		img = tf.image.resizeBilinear(img, [img.shape[0] * scale, img.shape[1] * scale]).div(255)
		let edges = cannyEdgeDetector(img)
		lines = RANSAC(edges, 21)
		let imgBuf = img.bufferSync()
		for(let i = 0; i < lines.length; i++) {
			let c = (i > 1 ? [0, 1, 0] : [0, 0, 1])
			drawLine(imgBuf, lines[i], c)
		}

		lines = [lines[0], lines[1]]
		setImg(img, "image")
		imgSize = img.shape
		// let groups = splitLines(lines)
		//
		// if(groups == null) {
		// 	console.log("FAILURE")
		// 	return
		// }
		//
		// let quads = getQuads(groups)

		let pi = Math.PI
		let pos = [0, 0, -5];
		let angle = [pi, pi+pi/4, pi/6]

		cube = new Cube(pos, angle, 1000, [img.shape[1]/2, img.shape[0]/2])
		showCube()
		document.getElementById('x').addEventListener('input', (event) => {
			pos[0] = event.target.valueAsNumber
			setErrorText()
		})

		document.getElementById('y').addEventListener('input', (event) => {
			pos[1] = -event.target.valueAsNumber
			setErrorText()
		})

		document.getElementById('z').addEventListener('input', (event) => {
			pos[2] = -event.target.valueAsNumber
			setErrorText()
		})

		document.getElementById('a').addEventListener('input', (event) => {
			angle[0] = event.target.valueAsNumber
			cube.dirty = true
			setErrorText()
		})

		document.getElementById('b').addEventListener('input', (event) => {
			angle[1] = event.target.valueAsNumber
			cube.dirty = true
			setErrorText()
		})

		document.getElementById('c').addEventListener('input', (event) => {
			angle[2] = event.target.valueAsNumber
			cube.dirty = true
			setErrorText()
		})

		document.getElementById('f').addEventListener('input', (event) => {
			cube.f = event.target.valueAsNumber
			setErrorText()
		})
	}
	im.src = "../imgs/cube_" + image_num + ".jpg";
}

function setErrorText() {
	let error = document.getElementById("error")
	error.innerHTML = "Error: " + cube.getTotalError(lines) + "</br>"
	let posDer = cube.getErrorDerivativeWRTPosition(lines)
	for(let p of posDer)
		error.innerHTML += p + "</br>"
	let angleDer = cube.getErrorDerivativeWRTAngle(lines)
	for(let a of angleDer)
		error.innerHTML += a + "</br>"
	let fDer = cube.getErrorDerivativeWRTFocalLength(lines)
	error.innerHTML += fDer + "</br>"
}

function showCube() {
	let drawing = tf.zeros([imgSize[0], imgSize[1], 4])
	cube.drawOnImage(drawing.bufferSync(), [0, 0, 0, 1])
	setImg(drawing, "draw");
}

function step() {
	cube.stepParameters(lines, 0.00003)
	setErrorText()
	showCube()
}

function setImg(img, canvasName) {
	img = normalise(img);
	let canvas = document.getElementById(canvasName);
	tf.browser.toPixels(img, canvas).then(() => tf.dispose(img))
}

test(2)

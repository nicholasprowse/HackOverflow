
function drawEdges(img, edges) {
	img = img.bufferSync();
	for(let i = 0; i < edges.length; i++) {
		img.set(1, edges[i][1], edges[i][0], 0)
		img.set(0, edges[i][1], edges[i][0], 1)
		img.set(0, edges[i][1], edges[i][0], 2)
	}
	return img
}

// Warning: Divide by zero errors are not handled
function drawLine(img, line, c) {
	const m = -line[0] / line[1];
	if(Math.abs(m) > 1) {
		// Iterate over y
		for(let y = 0; y < img.shape[0]; y++) {
			const x = -Math.round((line[1]*y + line[2]) / line[0])
			if(x >= 0 && x < img.shape[1])
				for(let i = 0; i < c.length; i++)
					img.set(c[i], y, x, i)

		}
	} else {
		// Iterate over x
		for(let x = 0; x < img.shape[1]; x++) {
			const y = -Math.round((line[0]*x + line[2]) / line[1])
			if(y >= 0 && y < img.shape[0])
				for(let i = 0; i < c.length; i++)
					img.set(c[i], y, x, i)

		}
	}

	return img
}

function drawLineSegment(img, p1, p2, c) {
	let line = cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
	const m = -line[0] / line[1];
	if(Math.abs(m) > 1) {
		// Iterate over y
		let sy = Math.min(p1[1], p2[1]), fy = Math.max(p1[1], p2[1])
		for(let y = Math.floor(sy); y <= fy; y++) {
			const x = -Math.round((line[1]*y + line[2]) / line[0])
			if(x >= 0 && x < img.shape[1]) {
				for(let i = 0; i < c.length; i++)
					img.set(c[i], y, x, i)
			}
		}
	} else {
		// Iterate over x
		let sx = Math.min(p1[0], p2[0]), fx = Math.max(p1[0], p2[0])
		for(let x = Math.floor(sx); x <= fx; x++) {
			const y = -Math.round((line[0]*x + line[2]) / line[1])
			if(y >= 0 && y < img.shape[0]) {
				for(let i = 0; i < c.length; i++)
					img.set(c[i], y, x, i)
			}
		}
	}

	return img
}

function drawQuad(img, q, c) {
	let corners = new Array(4)
	for(let i = 0; i < 4; i++)
		corners[i] = intersection(q[i], q[(i+1) % 4])

	for(let i = 0; i < 4; i++)
		img = drawLineSegment(img, corners[i], corners[(i+1)%4], c)
	return img
}

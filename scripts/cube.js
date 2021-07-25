// This just represents 3 faces of a cube, as a photo of a cube can only ever see 3 faces
class Cube {

	static points = [
		[0, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
		[1, 1, 0],
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1/3],
		[0, 1, 1/3],
		[1, 0, 1/3],
		[0, 0, 2/3],
		[0, 1, 2/3],
		[1, 0, 2/3],
		[0, 1/3, 0],
		[1, 1/3, 0],
		[0, 1/3, 1],
		[0, 2/3, 0],
		[1, 2/3, 0],
		[0, 2/3, 1],
		[1/3, 0, 0],
		[1/3, 1, 0],
		[1/3, 0, 1],
		[2/3, 0, 0],
		[2/3, 1, 0],
		[2/3, 0, 1]
	]

	static lines = [
		[0, 1],
		[0, 2],
		[0, 3],
		[1, 4],
		[1, 5],
		[2, 4],
		[2, 6],
		[3, 5],
		[3, 6],
		[7, 8],
		[7, 9],
		[10, 11],
		[10, 12],
		[13, 14],
		[13, 15],
		[16, 17],
		[16, 18],
		[19, 20],
		[19, 21],
		[22, 23],
		[22, 24]
	]

	constructor(pos, angle) {
		this.pos = pos
		this.angle = angle
	}

	transformPoint(p) {
		// Shorthand for cos and sin of each angle
		let [a, b, c] = this.angle
		let [ca, cb, cc] = [Math.cos(a), Math.cos(b), Math.cos(c)]
		let [sa, sb, sc] = [Math.sin(a), Math.sin(b), Math.sin(c)]
		// Roll Pitch Yaw matrix
		const e = [
			[ca * cb, -cb * sa, sb],
			[cc * sa + ca * sb * sc, ca * cc - sa * sb * sc, -cb * sc],
			[-ca * cc * sb + sa * sc, cc * sa * sb + ca * sc, cb * cc]
		]

		const transform = [0, 0, 0]
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++)
				transform[i] += e[i][j] * p[j]
			transform[i] += this.pos[i]
		}

		return transform
	}

	static projectPoint(p, f, u) {
		return [f * p[0] / p[2] + u[0], f * p[1] / p[2] + u[1]]
	}

	getTotalError(lines, f, u) {
		const p = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			p[i] = Cube.projectPoint(
				this.transformPoint(
					Cube.points[i]), f, u)

		let error = 0
		for(let line of Cube.lines) {
			let [u, v] = [p[line[0]], p[line[1]]]
			u = [u[0], u[1], 1]
			v = [v[0], v[1], 1]
			let min = Infinity
			for(let l of lines) {
				let num = Math.sqrt(dot(sub(u, v), sub(u, v))) * (dot(l, v)**3 - dot(l, u)**3)
				let denom = dot([l[0], -l[1], l[2]], sub(v, u))**2 * dot(l, sub(v, u))

				min = Math.min(min, num / denom)
			}
			error += min
		}
		return error
		//((u - v).(u - v) ((L.v)^3 - (L.u)^3)) / (3 ((L*{1, -1, 1}).(v - u))^2 (L.(v - u)))
	}

	drawOnImage(img, f, c) {
		const projectedPoints = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			projectedPoints[i] = Cube.projectPoint(
				this.transformPoint(
					Cube.points[i]), f, [img.shape[1] / 2, img.shape[0] / 2])

		for (let line of Cube.lines)
			img = drawLineSegment(img, projectedPoints[line[0]], projectedPoints[line[1]], c)

		return img
	}

}

function sub(a, b) {
	let diff = new Array(a.length);
	for(let i = 0; i < a.length; i++)
		diff[i] = a[i] - b[i];
	return diff;
}

function dot(a, b) {
	let sum = 0;
	for(let i = 0; i < a.length; i++)
		sum += a[i] * b[i];
	return sum;
}

for(let c of Cube.points) {
	c[0] -= 0.5
	c[1] -= 0.5
	c[2] -= 0.5
}

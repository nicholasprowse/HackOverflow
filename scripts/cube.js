// This just represents 3 faces of a cube, as a photo of a cube can only ever see 3 faces
class Cube {

	static corners = [
		[0, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
		[1, 1, 0],
		[1, 0, 1],
		[0, 1, 1]
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
		[3, 6]
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

	drawOnImage(img, f, u) {
		const projectedPoints = new Array(Cube.corners.length)
		for (let i = 0; i < Cube.corners.length; i++)
			projectedPoints[i] = Cube.projectPoint(
				this.transformPoint(
					Cube.corners[i]), f, [img.shape[1] / 2, img.shape[0] / 2])

		for (let line of Cube.lines)
			img = drawLineSegment(img, projectedPoints[line[0]], projectedPoints[line[1]], [0, 0, 0])

		return img
	}

}

for(let c of Cube.corners) {
	c[0] -= 0.5
	c[1] -= 0.5
	c[2] -= 0.5
}

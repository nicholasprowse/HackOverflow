import * as THREE from '../resources/threejs/build/three.module.js';

let cube_state = "RYRWBOBGGBROWYYRGWGOYWORGWBYRYRYBGWOYWOGBRWOWBOYGGROBB"
//				 "012345678901234567890123456789012345678901234567890123"
let cube

const red = new THREE.Color(1, 0, 0);
const green = new THREE.Color(0, 1, 0);
const blue = new THREE.Color(0, 0, 1);
const orange = new THREE.Color(1, 0.5, 0);
const white = new THREE.Color(1, 1, 1);
const yellow = new THREE.Color(1, 1, 0);
const black = new THREE.Color(0, 0, 0);
const magenta = new THREE.Color(1, 0, 1);
const c = {"W": white, "O": orange, "R": red, "B": blue, "Y": yellow, "G": green}
const boxSize = 0.95
const cubies = []
let rotating = false

window.setRotate = setRotate;

function main() {
	const canvas = document.querySelector('#c');
	const renderer = new THREE.WebGLRenderer({
		canvas
	});
	const fov = 75;
	const aspect = 2; // the canvas default
	const near = 0.1;
	const far = 5;
	const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
	camera.position.z = 4;

	const scene = new THREE.Scene();

	{
		const color = 0xFFFFFF;
		const intensity = 1;
		const light = new THREE.DirectionalLight(color, intensity);
		light.position.set(-1, 2, 4);
		scene.add(light);
	}

	createRubiksCube(scene)
	renderer.render(scene, camera);

	function render(time) {
		time *= 0.001; // convert time to seconds
		if(rotating) {
			for (let cube of cubies) {
				setRotationAboutAxis(cube, 0.01, 'x')
				setRotationAboutAxis(cube, 0.01, 'y')
			}
		}
		renderer.render(scene, camera);

		requestAnimationFrame(render);
	}

	requestAnimationFrame(render);
}

function setRotate() {
	rotating = !rotating
}

function setRotationAboutAxis(cube, angle, axis) {
	if(axis == 'x') {
		const x = cube.position.x
		cube.position.x = 0
    	cube.position.applyAxisAngle(new THREE.Vector3(1, 0, 0), angle); // rotate the POSITION
    	cube.position.x = x
		cube.rotateOnAxis(new THREE.Vector3(1, 0, 0), angle);
	}
	if(axis == 'z') {
		const z = cube.position.z
		cube.position.z = 0
    	cube.position.applyAxisAngle(new THREE.Vector3(0, 0, 1), angle); // rotate the POSITION
    	cube.position.z = z
		cube.rotateOnAxis(new THREE.Vector3(0, 0, 1), angle);
	}
	if(axis == 'y') {
		const y = cube.position.y
		cube.position.y = 0
    	cube.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), angle); // rotate the POSITION
    	cube.position.y = y
		cube.rotateOnAxis(new THREE.Vector3(0, 1, 0), angle);
	}
}

function createRubiksCube(scene) {
	for(let x = -1; x <= 1; x++)
		for(let y = -1; y <= 1; y++)
			for(let z = -1; z <= 1; z++)
				createCubie(x, y, z)

	for(let cube of cubies)
		scene.add(cube)
}

// x, y, z range from -1 to 1
function createCubie(x, y, z) {
	let colors = new Array(6)
	if(x == 1) {
		colors[0] = c[cube_state[28 - y + 12*z]]
		colors[1] = black
	} else if(x == -1) {
		colors[0] = black
		colors[1] = c[cube_state[22 + y + 12*z]]
	} else {
		colors[0] = black
		colors[1] = black
	}

	if(y == 1) {
		colors[2] = c[cube_state[25 + x + 12*z]]
		colors[3] = black
	} else if(y == -1) {
		colors[2] = black
		colors[3] = c[cube_state[31 - x + 12*z]]
	} else {
		colors[2] = black
		colors[3] = black
	}

	if(z == 1) {
		colors[4] = c[cube_state[49 + x - 3*y]]
		colors[5] = black
	} else if(z == -1) {
		colors[4] = black
		colors[5] = c[cube_state[4 + x + 3*y]]
	} else {
		colors[4] = black
		colors[5] = black
	}
	let cube = createCube(colors)
	cube.position.x = x
	cube.position.y = y
	cube.position.z = z
}

// RLUDFB
function createCube(colors) {
	let materials = new Array(6)
	for (let i = 0; i < 6; i++) {
		if(!colors[i])
			colors[i] = magenta
		materials[i] = new THREE.MeshPhongMaterial({
			color: colors[i]
		})
	}

	const geometry = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
	const cube = new THREE.Mesh(geometry, materials)
	cubies.push(cube)
	return cube
}

/*
   RYR
   WBO
   BGG
BROWYYRGWGOY
WORGWBYRYRYB
GWOYWOGBRWOW
   BOY
   GGR
   OBB

RYRWBOBGGBROWYYRGWGOYWORGWBYRYRYBGWOYWOGBRWOWBOYGGROBB
*/

main()

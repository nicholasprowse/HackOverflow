// Convert rubix cube in the following format
let input = new URLSearchParams(window.location.search).get('state')
let {PythonShell} = require('python-shell');

//path depnds on host machine that we will run for the demo or however it will be deployed
let options = {
	scriptPath: './',
	args: [input],
	pythonPath: '/opt/homebrew/bin/python3'
};

var test = new PythonShell('rubixSolver.py',options);
test.on('message', function(message) {
	console.log(message);
});

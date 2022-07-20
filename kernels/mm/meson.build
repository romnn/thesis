project('cuda dependency', 'cpp', 'cuda')

exe = executable('main', 'mm.cc', 'mm.cu', dependencies: dependency('cuda'))
test('cudatest', exe)

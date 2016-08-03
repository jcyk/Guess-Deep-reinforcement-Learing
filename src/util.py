from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import tanh,sigmoid
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

class Contextual_RNNCell(object):
	def __call__(self, inputs, context, state, scope=None):
		raise NotImplementedError("Abstract method")
	
	@property
	def state_size(self):
		raise NotImplementedError("Abstract method")
  	
  	@property
	def output_size(self):
		raise NotImplementedError("Abstract method")

	def zero_state(self, batch_size, dtype):
		state_size = self.state_size
		if nest.is_sequence(state_size):
			state_size_flat = nest.flatten(state_size)
			zeros_flat = [
          		array_ops.zeros(
              	array_ops.pack(rnn_cell._state_size_with_prefix(s, prefix=[batch_size])),
              	dtype=dtype)
          		for s in state_size_flat]
			for s, z in zip(state_size_flat, zeros_flat):
				z.set_shape(rnn_cell._state_size_with_prefix(s, prefix=[None]))
			zeros = nest.pack_sequence_as(structure=state_size,flat_sequence=zeros_flat)
		else:
			zeros_size = rnn_cell._state_size_with_prefix(state_size, prefix=[batch_size])
			zeros = array_ops.zeros(array_ops.pack(zeros_size), dtype=dtype)
			zeros.set_shape(rnn_cell._state_size_with_prefix(state_size, prefix=[None]))
		return zeros

class Contextual_GRUCell(Contextual_RNNCell):
	"""Contextual Gated Recurrent Unit cell (https://arxiv.org/abs/1602.06291)."""
	def __init__(self, num_units, activation=tanh):
		self._num_units = num_units
		self._activation = activation
	
	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units


	def __call__(self, inputs, context, state, scope=None):
		"""Contextual Gated recurrent unit (CGRU) with nunits cells."""
		with vs.variable_scope(scope or type(self).__name__):
			with vs.variable_scope("Gates"):
				r, u = array_ops.split(1, 2, rnn_cell._linear([inputs, context, state],
                                             2 * self._num_units, True, 1.0))
				r, u = sigmoid(r), sigmoid(u)
			with vs.variable_scope("Candidate"):
				c = self._activation(rnn_cell._linear([inputs, context, r * state],
                                     self._num_units, True))
			new_h = u * state + (1 - u) * c
		return new_h, new_h

# pylint: disable=unused-argument
def _rnn_step(
	time, sequence_length, min_sequence_length, max_sequence_length,
	zero_output, state, call_cell, state_size, skip_conditionals=False):

	# Convert state to a list for ease of use
	flat_state = nest.flatten(state)
	flat_zero_output = nest.flatten(zero_output)

	def _copy_one_through(output, new_output):
		copy_cond = (time >= sequence_length)
		return math_ops.select(copy_cond, output, new_output)

	def _copy_some_through(flat_new_output, flat_new_state):
		flat_new_output = [
        	_copy_one_through(zero_output, new_output)
        	for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
		flat_new_state = [
        	_copy_one_through(state, new_state)
        	for state, new_state in zip(flat_state, flat_new_state)]
		return flat_new_output + flat_new_state

	def _maybe_copy_some_through():
		"""Run RNN step.  Pass through either no or some past state."""
		new_output, new_state = call_cell()

		nest.assert_same_structure(state, new_state)

		flat_new_state = nest.flatten(new_state)
		flat_new_output = nest.flatten(new_output)
		return control_flow_ops.cond(
        	# if t < min_seq_len: calculate and return everything
        	time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        	# else copy some of it through
        	lambda: _copy_some_through(flat_new_output, flat_new_state))

	# TODO(ebrevdo): skipping these conditionals may cause a slowdown,
	# but benefits from removing cond() and its gradient.  We should
	# profile with and without this switch here.
	if skip_conditionals:
		# Instead of using conditionals, perform the selective copy at all time
		# steps.  This is faster when max_seq_len is equal to the number of unrolls
		# (which is typical for dynamic_rnn).
		new_output, new_state = call_cell()
		nest.assert_same_structure(state, new_state)
		new_state = nest.flatten(new_state)
		new_output = nest.flatten(new_output)
		final_output_and_state = _copy_some_through(new_output, new_state)
	else:
		empty_update = lambda: flat_zero_output + flat_state
		final_output_and_state = control_flow_ops.cond(
        	# if t >= max_seq_len: copy all state through, output zeros
        	time >= max_sequence_length, empty_update,
        	# otherwise calculation is required: copy some or all of it through
        	_maybe_copy_some_through)

	if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
		raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
	final_output = final_output_and_state[:len(flat_zero_output)]
	final_state = final_output_and_state[len(flat_zero_output):]

	for output, flat_output in zip(final_output, flat_zero_output):
		output.set_shape(flat_output.get_shape())
	for substate, flat_substate in zip(final_state, flat_state):
		substate.set_shape(flat_substate.get_shape())

	final_output = nest.pack_sequence_as(
		structure=zero_output, flat_sequence=final_output)
	final_state = nest.pack_sequence_as(
		structure=state, flat_sequence=final_state)

	return final_output, final_state

def contextual_rnn(cell, inputs, context, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
	if not isinstance(cell, Contextual_RNNCell):
		raise TypeError("cell must be an instance of Contextual_RNNCell")

	# By default, time_major==False and inputs are batch-major: shaped
	#   [batch, time, depth]
	# For internal calculations, we transpose to [time, batch, depth]
	flat_input = nest.flatten(inputs)

	if not time_major:
		# (B,T,D) => (T,B,D)
		flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                       for input_ in flat_input)

	parallel_iterations = parallel_iterations or 32
	if sequence_length is not None:
		sequence_length = math_ops.to_int32(sequence_length)
		sequence_length = array_ops.identity(  # Just to find it in the graph.
        	sequence_length, name="sequence_length")

	# Create a new scope in which the caching device is either
	# determined by the parent scope, or is set to place the cached
	# Variable using the same placement as for the rest of the RNN.
	with vs.variable_scope(scope or "RNN") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)
		input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
		batch_size = input_shape[0][1]

		for input_ in input_shape:
			if input_[1].get_shape() != batch_size.get_shape():
				raise ValueError("All inputs should have the same batch size")

		if initial_state is not None:
			state = initial_state
		else:
			if not dtype:
				raise ValueError("If no initial_state is provided, dtype must be.")
			state = cell.zero_state(batch_size, dtype)

		def _assert_has_shape(x, shape):
			x_shape = array_ops.shape(x)
			packed_shape = array_ops.pack(shape)
			return logging_ops.Assert(
          	math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          	["Expected shape for Tensor %s is " % x.name,
           	packed_shape, " but saw shape: ", x_shape])

		if sequence_length is not None:
			# Perform some shape validation
			with ops.control_dependencies(
          	[_assert_has_shape(sequence_length, [batch_size])]):
				sequence_length = array_ops.identity(sequence_length, name="CheckSeqLen")

		inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

		(outputs, final_state) = _contextual_rnn_loop(
        	cell,
        	inputs,
        	context,
        	state,
        	parallel_iterations=parallel_iterations,
        	swap_memory=swap_memory,
        	sequence_length=sequence_length,
        	dtype=dtype)

		# Outputs of _contextual_rnn_loop are always shaped [time, batch, depth].
		# If we are performing batch-major calculations, transpose output back
		# to shape [batch, time, depth]
		if not time_major:
			# (T,B,D) => (B,T,D)
			flat_output = nest.flatten(outputs)
			flat_output = [array_ops.transpose(output, [1, 0, 2])
                     for output in flat_output]
			outputs = nest.pack_sequence_as(
          		structure=outputs, flat_sequence=flat_output)

		return (outputs, final_state)


def _contextual_rnn_loop(cell,
                      inputs,
                      context,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
	state = initial_state
	assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

	state_size = cell.state_size

	flat_input = nest.flatten(inputs)
	flat_output_size = nest.flatten(cell.output_size)

	# Construct an initial output
	input_shape = array_ops.shape(flat_input[0])
	time_steps = input_shape[0]
	batch_size = input_shape[1]

	inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

	const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

	for shape in inputs_got_shape:
		if not shape[2:].is_fully_defined():
			raise ValueError(
         	 "Input size (depth of inputs) must be accessible via shape inference,"
          	" but saw value None.")
		got_time_steps = shape[0]
		got_batch_size = shape[1]
		if const_time_steps != got_time_steps:
			raise ValueError(
          	"Time steps is not the same for all the elements in the input in a "
          	"batch.")
		if const_batch_size != got_batch_size:
			raise ValueError(
			"Batch_size is not the same for all the elements in the input.")

	# Prepare dynamic conditional copying of state & output
	def _create_zero_arrays(size):
		size = rnn_cell._state_size_with_prefix(size, prefix=[batch_size])
		return array_ops.zeros(
        	array_ops.pack(size), _infer_state_dtype(dtype, state))

	flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
	zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

	if sequence_length is not None:
		min_sequence_length = math_ops.reduce_min(sequence_length)
		max_sequence_length = math_ops.reduce_max(sequence_length)

	time = array_ops.constant(0, dtype=dtypes.int32, name="time")

	with ops.op_scope([], "contextual_rnn") as scope:
		base_name = scope

	def _create_ta(name, dtype):
		return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        tensor_array_name=base_name + name)

	output_ta = tuple(_create_ta("output_%d" % i,
                               _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
	input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                   for i in range(len(flat_input)))

	input_ta = tuple(ta.unpack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

	def _time_step(time, output_ta_t, state):

		input_t = tuple(ta.read(time) for ta in input_ta)
		# Restore some shape information
		for input_, shape in zip(input_t, inputs_got_shape):
			input_.set_shape(shape[1:])

		input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
		call_cell = lambda: cell(input_t, context, state)

		if sequence_length is not None:
			(output, new_state) = _rnn_step(
          	time=time,
          	sequence_length=sequence_length,
          	min_sequence_length=min_sequence_length,
          	max_sequence_length=max_sequence_length,
          	zero_output=zero_output,
          	state=state,
          	call_cell=call_cell,
          	state_size=state_size,
          	skip_conditionals=True)
		else:
			(output, new_state) = call_cell()

		# Pack state if using state tuples
		output = nest.flatten(output)

		output_ta_t = tuple(
        	ta.write(time, out) for ta, out in zip(output_ta_t, output))

		return (time + 1, output_ta_t, new_state)

	_, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

	# Unpack final output if not using output tuples.
	final_outputs = tuple(ta.pack() for ta in output_final_ta)

	# Restore some shape information
	for output, output_size in zip(final_outputs, flat_output_size):
		shape = rnn_cell._state_size_with_prefix(
        	output_size, prefix=[const_time_steps, const_batch_size])
		output.set_shape(shape)

	final_outputs = nest.pack_sequence_as(
		structure=cell.output_size, flat_sequence=final_outputs)

	return (final_outputs, final_state)


def _infer_state_dtype(explicit_dtype, state):
	if explicit_dtype is not None:
		return explicit_dtype
	elif nest.is_sequence(state):
		inferred_dtypes = [element.dtype for element in nest.flatten(state)]
		if not inferred_dtypes:
			raise ValueError("Unable to infer dtype from empty state.")
		all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
		if not all_same:
			raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
		return inferred_dtypes[0]
	else:
		return state.dtype

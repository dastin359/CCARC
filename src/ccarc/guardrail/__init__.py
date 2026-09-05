"""What stops the agent reaching anything it is not meant to reach.

Four layers, each granting back exactly what the one below took away:

* :mod:`confine` removes the filesystem outside the run's own workspace.
* :mod:`network` removes the network, leaving one reachable destination.
* :mod:`egress_proxy` grants back the hosts model inference needs, and
  refuses every other.
* :mod:`arc_proxy` grants back the game traffic, holding the API key the
  solver never sees and filtering the human baseline out of every response.

The first two make the score honest by construction. The second two are what
let the agent play at all.
"""

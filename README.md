# belief_propagation_attack

An implementation of the Belief Propagation Attack in python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Use

```
make install
```

to install necessary dependencies from `REQUIREMENTS.txt`.

### Building

If you have `cython` installed, we recommend you run

```
make cython-build
```

Otherwise, you can build from the provided `.c` files:

```
make build
```

Building may take a while! To check whether it has built correctly, run

```
python belief_propagation_attack/main.py
```

If this simulates an attack, you're good to go.

## Running the tests

WIP: run

```
python TestUtility.py
```

and see what happens.

## Built With

* [Python 2.7](https://www.python.org/download/releases/2.7/) - The language used
* [networkx](https://networkx.github.io/) - Used to simulate the factor graph
* [cython](https://cython.org/) - To speed up some of the bottlenecked functions (`arrayXOR`)

## Contributing

<!-- Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. -->
Please don't. I'll update this once I've finished my PhD.

## Authors

* **Joey Green** - *All work* - [JustJoeyGreen](https://github.com/JustJoeyGreen)

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Elisabeth Oswald for my PhD supervision
* Arnab Roy for suggesting some of the graph reductions
* [xkcd](https://xkcd.com/353/) for helping me remain sane through my PhD life

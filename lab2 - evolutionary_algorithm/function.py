from dataclasses import dataclass, field


@dataclass
class Function:
    function: callable
    domain: list[tuple] = field(default_factory=list)
    number_of_variables: int = field(init=False)

    def __post_init__(self):
        self.number_of_variables = len(self.domain)

    def __call__(self, *args):
        return self.function(*args)

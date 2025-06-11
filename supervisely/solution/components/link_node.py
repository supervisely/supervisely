from supervisely.app.widgets import SolutionCard
from supervisely.solution.base_node import SolutionCardNode, SolutionElement


class LinkNode(SolutionElement):
    """
    Base class for nodes that can be used as links to external resources.
    """

    def __init__(
        self,
        title: str,
        description: str,
        link: int,
        width: int = 250,
        x: int = 0,
        y: int = 0,
        *args,
        **kwargs,
    ):
        """
        Initialize the Manual Import GUI widget.

        :param link: URL to the external resource.
        """
        self.link = link
        self.title = title
        self.description = description
        self.width = width
        self.card = self._create_card()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        super().__init__(*args, **kwargs)

    def _create_card(self) -> SolutionCard:
        """
        Creates and returns the SolutionCard for the Manual Import widget.
        """
        return SolutionCard(
            title=self.title,
            tooltip=self._create_tooltip(),
            width=self.width,
            tooltip_position="right",
            link=self.link,
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        """
        Creates and returns the tooltip for the Manual Import widget.
        """
        return SolutionCard.Tooltip(description=self.description)

import torch

from tqdm import tqdm
from typing import Optional, Callable
from torchvision import transforms
from logging import getLogger

logger = getLogger(__name__)


class ProjectedGradientDescent:
    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        inverse_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        """Initializes the Projected Gradient Descent attack.

        :param model: The model to attack.
        :param epsilon: The maximum perturbation allowed.
        :param alpha: The step size
        :param transform: The transformation to apply to the image
        :param inverse_transform: The inverse transformation to
            apply to the image
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

        self.transform = (
            transform
            if transform is not None
            else transforms.Lambda(lambda x: x)
        )
        self.inverse_transform = (
            inverse_transform
            if inverse_transform is not None
            else transforms.Lambda(lambda x: x)
        )

        # construct pixel projection
        self.pixel_projection = transforms.Compose(
            [
                self.inverse_transform,
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )

    @staticmethod
    def _assert_ranges(image: torch.Tensor) -> None:
        """Asserts that the pixel values are in the correct range.

        :param image: The image to check
        """
        if not (0 <= image).all() or not (image <= 1).all():
            raise ValueError("Pixel values must be in the range [0, 1]")

    @staticmethod
    def _add_batch_dimension(image: torch.Tensor) -> torch.Tensor:
        """Adds a batch dimension to the image.

        :param image: The image to add the batch dimension to
        :return: The image with a batch dimension
        """
        if image.dim() == 3:
            return image.unsqueeze(0)

        if image.dim() == 4:
            if image.size(0) != 1:
                raise ValueError("Batch size must be 1")
            else:
                return image

        raise ValueError("Image must have 3 or 4 dimensions")

    def generate_image(
        self,
        image: torch.Tensor,
        target_class: int,
        max_num_steps: int = 100,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generates an adversarial image using the
        Projected Gradient Descent attack.

        :param image: The original image
        :param target_class: The target
        :param max_num_steps: The maximum number of steps to run the attack
        :param verbose: Whether to print progress bar
        :return: The adversarial image
        """
        ProjectedGradientDescent._assert_ranges(image)
        image = ProjectedGradientDescent._add_batch_dimension(image)

        image = self.transform(image)
        adv_image = image.clone()

        self.model.eval()
        success = False

        for _ in tqdm(range(max_num_steps), disable=not verbose):
            # pass the adversarial image through the model
            adv_image = adv_image.detach().requires_grad_()
            output = self.model(adv_image)

            if torch.argmax(output[0]).item() == target_class:
                success = True
                break

            # compute the gradient of the target class
            loss = output[0, target_class]
            loss.backward()

            # update the adversarial image
            gradients = adv_image.grad
            adv_image = adv_image + self.alpha * gradients.sign()  # type: ignore[union-attr]  # noqa: E501

            # projection image to ensure perturbation stays
            # within the epsilon ball
            perturbation = torch.clamp(
                adv_image - image, min=-self.epsilon, max=self.epsilon
            )
            adv_image = image + perturbation

            # project into the valid pixel range
            adv_image = self.pixel_projection(adv_image)
            adv_image = self.transform(adv_image)

            # reset the gradients
            self.model.zero_grad()

        if not success:
            logger.info(
                "Failed to generate adversarial image. Consider increasing "
                "the number of steps or tune the other parameters."
            )

        adv_image = self.inverse_transform(adv_image)
        return adv_image.squeeze(0).detach()

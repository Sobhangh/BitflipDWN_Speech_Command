from torch.utils.data import Dataset 
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import random, math
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch

from abc import ABC, abstractmethod
from typing_extensions import override

class dSprites(Dataset, ABC):
    def __init__(
        self, 
        data: dict = None,
        shapes: list[str] =['square', 'ellipse', 'triangle', 'heart'] , 
        resolution: tuple[int, int] = (128, 128),
        transformation = None
    ):
        self.shapes = shapes 
        self.resolution = resolution 
        self.data = data
        self.transformation = transformation
    
    def select_radii(
        self, 
        n: int, 
        min_ratio: float = 0.05, 
        max_ratio: float = 0.15,
    ) -> list[int]:
        """
        Selects radii for n non-overlapping shapes within the image.
        Dynamically adjusts the maximum radius based on n to ensure they can fit.
        """
        base = min(self.resolution)
        
        # Calculate a safe maximum ratio based on a grid layout.
        # For n shapes, a safe grid size is ceil(sqrt(n)).
        grid_size = math.ceil(math.sqrt(n))
        
        # The maximum allowed radius ratio to guarantee they can physically fit
        safe_max_ratio = 1.0 / (2.0 * grid_size)
        
        # Apply the bounding
        actual_max_ratio = min(max_ratio, safe_max_ratio)
        
        # Ensure min_ratio doesn't exceed max_ratio
        actual_min_ratio = min(min_ratio, actual_max_ratio * 0.75) 
        
        return [int(random.uniform(base * actual_min_ratio, base * actual_max_ratio)) for _ in range(n)]

    def select_coordinates(
        self, 
        radii: list[int], 
        padding: int = 8,
        max_attempts_per_shape: int = 500,
        max_global_attempts: int = 200
    ) -> list[tuple[int, int]]:
        """
        Selects non-overlapping coordinates iteratively, avoiding recursion limits.
        
        Args:
            radii: List of shape radii.
            padding: Padding pixels between objects and borders.
            max_attempts_per_shape: How many times to try placing a single shape.
            max_global_attempts: How many times to restart the entire layout.
        """
        # Try to build the whole layout up to max_global_attempts times
        for _ in range(max_global_attempts):
            coordinates = []
            layout_success = True
            
            for r in radii:
                shape_placed = False
                
                # Try to place the current shape up to max_attempts_per_shape times
                for _ in range(max_attempts_per_shape):
                    cx = random.randint(r + padding, self.resolution[1] - r - padding)
                    cy = random.randint(r + padding, self.resolution[0] - r - padding)
                    
                    overlap = False
                    for i, (ox, oy) in enumerate(coordinates):
                        or_ = radii[i]
                        dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
                        if dist < (r + or_ + padding):
                            overlap = True
                            break
                    
                    if not overlap:
                        coordinates.append((cx, cy))
                        shape_placed = True
                        break # Move on to the next radius
                
                # If we failed to place this shape after 100 tries, abort this layout attempt
                if not shape_placed:
                    layout_success = False
                    break 
            
            # If all shapes were successfully placed, return the coordinates
            if layout_success:
                return coordinates
                
        # If we exit both loops, the layout is too densely packed to solve randomly
        raise RuntimeError("Failed to find non-overlapping coordinates. Decrease radii or padding.")

    def display(self, idx):
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        for i, panel in enumerate(self.data['samples'][idx]):
            ax[i].imshow(panel)
        ax[self.data['o3'][idx]].set_title('ODD')
        plt.show()

    @classmethod
    def from_path(cls, 
                  path: str, 
                  resolution: tuple[int, int] = (128, 128), 
                  transforms = None, 
                  **kwargs):
        f = np.load(path, allow_pickle=True)
        data = {
            'samples': torch.from_numpy(f['samples']),  # (N, 3, H, W)
            'o3': torch.from_numpy(f['meta_data'].item()['o3']) # (N)
        }
        return cls(data=data, resolution=resolution, transformation=transforms, **kwargs)

    def __len__(self):
        return len(self.data['samples'])

    def __getitem__(self, index):
        x = self.transformation(self.data['samples'][index]) if self.transformation is not None else self.data['samples'][index]
        y = self.data['o3'][index]
        return x.float(), y.long()

    @abstractmethod
    def precompute(self, n: int, path: str):
        raise NotImplementedError
        

class CountingShapes(dSprites):

    def create_panel(self, shape: str, n: int):
        radii = self.select_radii(n) # select radii
        coords = self.select_coordinates(radii) # select coordinates
        img = Image.new('L', self.resolution, 'black')
        draw = ImageDraw.Draw(img)
        for (cx, cy), r in zip(coords, radii):
            if shape == 'square':
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill='white')
            elif shape == 'ellipse':
                rx = r
                ry = int(r * 0.6)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill='white')
            elif shape == 'triangle':
                points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r),]
                draw.polygon(points, fill='white')
            elif shape == 'heart':
                points = []
                for i in range(360):
                    angle = math.radians(i)
                    x = r * 16 * math.sin(angle) ** 3
                    y = -r * (13 * math.cos(angle) - 5 * math.cos(2*angle) 
                            - 2 * math.cos(3*angle) - math.cos(4*angle))
                    points.append((cx + x / 17, cy + y / 17))
                draw.polygon(points, fill='white')
        return img
    

    def sample(self):
        n, out = np.random.choice(range(1,6), size=2, replace=False).tolist()
        counts = [n, n, out]
        random.shuffle(counts)
        shapes = np.random.choice(self.shapes, size=3, replace=False).tolist()
        panels = tuple(map(self.create_panel, shapes, counts))
        metadata = {'shapes': shapes, 'counts': counts, 'o3': counts.index(out)}
        return panels, metadata
    
    @override
    def precompute(self, n, path):
        samples = np.empty((n, 3, *self.resolution), dtype=np.bool_)
        meta_data = dict(shapes = [], 
                         counts = np.zeros((n, 3), dtype=np.int8),
                         o3 = np.ones(n, dtype=np.int8) * -1)
        
        for i in tqdm(range(n), desc=f'{self.__class__.__name__}-precompute'):
            panels, meta_panels = self.sample()
            samples[i] = np.stack([np.array(p, dtype=bool) for p in panels]) 
            meta_data['shapes'].append(meta_panels['shapes'])
            meta_data['counts'][i] = meta_panels['counts']
            meta_data['o3'][i] = meta_panels['o3']

        np.savez_compressed(
            path,
            samples=samples,                          # (N, 3, H, W) bool
            meta_data=meta_data
        )

class OrientationShapes(CountingShapes):

    def __init__(self, num_objs=1, **kwargs):
        super().__init__(**kwargs)
        self.num_objects = num_objs

    def rotate(
        self,
        center: tuple[float, float], 
        points: list[tuple[int, int]], 
        angle_deg: float
    ) -> list[tuple[int, int]]:
        """Rotates a list of points around a center.
        
        Args:
            center (tuple[int, int]): Rotation center.
            points (list[tuple[int, int]]): List of points of the figure.
            angle_deg (float): Rotation degrees.

        Returns:
            list[tuple[int, int]]: List of rotated points.
        """
        cx, cy = center 
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = []
        for x, y in points:
            x, y = x - cx, y - cy
            rotated.append((
                cx + x * cos_a - y * sin_a,
                cy + x * sin_a + y * cos_a
            ))
        return rotated
        
    def create_panel(self, shape: str, degree: int):
        """Creates panel with a specific shape and orientation degree."""
        radii = self.select_radii(n=self.num_objects)
        coords = self.select_coordinates(radii, padding=10)
        img = Image.new('L', self.resolution, 'black')
        draw = ImageDraw.Draw(img)
        degrees = np.zeros(self.num_objects, dtype=np.int32)
        degrees[0] = degree

        for (cx, cy), r, d in zip(coords, radii, degrees):
            if shape == 'square':
                points = [
                    (cx - r, cy - r), (cx + r, cy - r),
                    (cx + r, cy + r), (cx - r, cy + r)
                ]
                draw.polygon(self.rotate((cx, cy), points, d), fill='white')

            elif shape == 'ellipse':
                # draw on a temp image centered at origin, then paste
                tmp = Image.new('L', self.resolution, 'black')
                tmp_draw = ImageDraw.Draw(tmp)
                rx, ry = r, int(r * 0.6)
                tmp_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill='white')
                tmp = tmp.rotate(d, center=(cx, cy))
                img = Image.composite(tmp, img, tmp)
                draw = ImageDraw.Draw(img)  # refresh draw handle after composite

            elif shape == 'triangle':
                points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
                draw.polygon(self.rotate((cx,cy), points, d), fill='white')

            elif shape == 'heart':
                points = []
                for i in range(360):
                    angle = math.radians(i)
                    x = r * 16 * math.sin(angle) ** 3
                    y = -r * (13 * math.cos(angle) - 5 * math.cos(2*angle)
                              - 2 * math.cos(3*angle) - math.cos(4*angle))
                    points.append((cx + x / 17, cy + y / 17))
                draw.polygon(self.rotate((cx,cy), points, d), fill='white')

        return img
    
    def sample(self):
        # select 3 different shapes
        shapes = np.random.choice(self.shapes, size=3, replace=False).tolist()

        # select random orientation
        if shapes[0] in ['square']:
            degree = np.random.randint(0, 90, 1).item()
        elif shapes[0] in ['ellipse']:
            degree = np.random.randint(0, 120, 1).item()
        elif shapes[0] == 'triangle':
            degree = np.random.randint(0, 180, 1).item()
        else:
            degree = np.random.randint(0, 360, 1).item()
        panels = list(map(self.create_panel, shapes, (degree, 0, 0)))
        perm = np.random.permutation(3)
        panels = [panels[i] for i in perm]
        shapes = [shapes[i] for i in perm]
        odd_index = np.where(perm == 0)[0][0]
        metadata = {'shapes': shapes, 'degree': degree, 'o3': odd_index}
        return panels, metadata
    
    @override
    def precompute(self, n, path):
        samples = np.empty((n, 3, *self.resolution), dtype=np.bool_)
        meta_data = dict(shapes = [], 
                         degree = np.zeros(n, dtype=np.int32) * -1,
                         o3 = np.ones(n, dtype=np.int8) * -1)
        
        for i in tqdm(range(n), desc=f'{self.__class__.__name__}-precompute'):
            panels, meta_panels = self.sample()
            samples[i] = np.stack([np.array(p, dtype=bool) for p in panels]) 
            meta_data['shapes'].append(meta_panels['shapes'])
            meta_data['degree'][i] = meta_panels['degree']
            meta_data['o3'][i] = meta_panels['o3']

        np.savez_compressed(
            path,
            samples=samples,                          # (N, 3, H, W) bool
            meta_data=meta_data
        )

class PairedShapes(CountingShapes):
       
    def create_panel(self, shapes: tuple[str, str]):
        """Creates panel with a two specific shapes."""
        radii = self.select_radii(2)
        coords = self.select_coordinates(radii, padding=10)
        img = Image.new('L', self.resolution, 'black')
        draw = ImageDraw.Draw(img)

        for (cx, cy), r, shape in zip(coords, radii, shapes):
            if shape == 'square':
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill='white')
            elif shape == 'ellipse':
                rx = r
                ry = int(r * 0.6)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill='white')
            elif shape == 'triangle':
                points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r),]
                draw.polygon(points, fill='white')
            elif shape == 'heart':
                points = []
                for i in range(360):
                    angle = math.radians(i)
                    x = r * 16 * math.sin(angle) ** 3
                    y = -r * (13 * math.cos(angle) - 5 * math.cos(2*angle) 
                            - 2 * math.cos(3*angle) - math.cos(4*angle))
                    points.append((cx + x / 17, cy + y / 17))
                draw.polygon(points, fill='white')
        return img

    def sample(self):
        shape1, shape2, out = np.random.choice(self.shapes, size=3, replace=False).tolist()
        panels = [
            self.create_panel([shape1, out]),       # odd one
            self.create_panel([shape1, shape2]),
            self.create_panel([shape1, shape2])
        ]
        shapes = [shape1, shape2, out]
        perm = np.random.permutation(3)
        panels = [panels[i] for i in perm]
        odd_index = np.where(perm == 0)[0][0]
        metadata = {'shapes': shapes, 'o3': odd_index}
        return panels, metadata
    
    @override
    def precompute(self, n, path):
        samples = np.empty((n, 3, *self.resolution), dtype=np.bool_)
        meta_data = dict(shapes = [], 
                         o3 = np.ones(n, dtype=np.int8) * -1)
        
        for i in tqdm(range(n), desc=f'{self.__class__.__name__}-precompute'):
            panels, meta_panels = self.sample()
            samples[i] = np.stack([np.array(p, dtype=bool) for p in panels]) 
            meta_data['shapes'].append(meta_panels['shapes'])
            meta_data['o3'][i] = meta_panels['o3']

        np.savez_compressed(
            path,
            samples=samples,                          # (N, 3, H, W) bool
            meta_data=meta_data
        )


if __name__ == '__main__':
    OrientationShapes().precompute(n=70000, path='o3-task/orientation-o3-128.npz')
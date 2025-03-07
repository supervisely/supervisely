from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.annotation.obj_class import ObjClass


from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh


type_to_zmdi_icon = {
    AnyGeometry: "zmdi zmdi-shape",
    Rectangle: "zmdi zmdi-crop-din",  # "zmdi zmdi-square-o"
    # Polygon: "icons8-polygon",  # "zmdi zmdi-edit"
    Bitmap: "zmdi zmdi-brush",
    AlphaMask: "zmdi zmdi-brush",
    Polyline: "zmdi zmdi-gesture",
    Point: "zmdi zmdi-dot-circle-alt",
    Cuboid: "zmdi zmdi-ungroup",  #
    # Cuboid3d: "zmdi zmdi-codepen",
    Pointcloud: "zmdi zmdi-cloud-outline",  #  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "zmdi zmdi-layers",  # "zmdi zmdi-collection-item"
    Point3d: "zmdi zmdi-filter-center-focus",  # "zmdi zmdi-select-all"
}

type_to_icons8_icon = {
    Polygon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAB6klEQVRYhe2Wuy8EURTGf+u5VESNXq2yhYZCoeBv8RcI1i6NVUpsoVCKkHjUGlFTiYb1mFmh2MiKjVXMudmb3cPOzB0VXzKZm5k53/nmvO6Ff4RHD5AD7gFP1l3Kd11AHvCBEpAVW2esAvWmK6t8l1O+W0lCQEnIJoAZxUnzNQNkZF36jrQjgoA+uaciCgc9VaExBOyh/6WWAi1VhbjOJ4FbIXkBtgkK0BNHnYqNKUIPeBPbKyDdzpld5T6wD9SE4AwYjfEDaXFeFzE/doUWuhqwiFsOCwqv2hV2lU/L+sHBscGTxdvSFVoXpAjCZdauMHVic6ndl6U1VBsJCFhTeNUU9IiIEo3qvQYGHAV0AyfC5wNLhKipXuBCjA5wT8WxcM1FMRoBymK44CjAE57hqIazwCfwQdARcXa3UXHuRXVucIjb7jYvNkdxBZg0TBFid7PQTRAtX2xOiXkuMAMqYwkIE848rZFbjyNAmw9bIeweaZ2A5TgC7PnwKkTPtN+cTOrsyN3FEWAjRTAX6sA5ek77gSL6+WHZVQDAIHAjhJtN78aAS3lXAXYIivBOnCdyOAUYB6o0xqsvziry7FLE/Cp20cNcJEjDr8MUmVOVRzkVN+Nd7vZGVXXgiwxtPiRS5WFhz4fEq/zv4AvToMn7vCn3eAAAAABJRU5ErkJggg==",
    # Polyline: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAYAAACOEfKtAAAABmJLR0QA/wD/AP+gvaeTAAAD2ElEQVR4nO2bvU8UURDAfwjSqFEiIRHODz6UwthoQ2JsNNFCsJFGYzB2XGyIxto/gGhjIyUx4gfQIEaIhQlG7YwaG0NnIyIfgpggemAx73KbY++Ovd27t3s7v2Szd+9u581O3ps3b3YWFEVRFEVRFEVRlPKSAIaBZXOMAoetahQhEsA8sJF1zANNFvWKDMOIwZ4hBmsCxk3bE4t6RYZlxFhNwBvgNbDftP3cqpBtJVEtGqyYcw2wjhhO8cBDxGgvgFbEJz43bcMW9YoMB4EU7otIm0W9PGEzjOhDDLYIfAeWjC6RMp6tMOIQ4gM3gM4S91UybIYRE6afoRL3U1ICCSOKoMf0MQc0+BVmM4xJhw1VbA4jdgMjwFWgPsA+G4C75vMNYDZA2WVnFDHaOOIPnWGE8/gHTAG3gHaffQ4ZmRM+5YSCI8g0yjbYLHASSCI3upr1+xegHzgFVHvor9Ncv4IsIhVBE7JgLJEx0NGs/+wCuoFBNht8zrR3m/85aUEWqgUkTFk01/SV4D5CwTsKhxXVyMjrR0ai05iryIhNIiPYbXSngAOlUd8+t5GbvOfhmnbEN04hvjLbYG4h0qPgVA4XHcgNThd5fT2yao+QMaBbiLTkW9OQUk1mZ9LqU1bapyYQ401RnhjTOo+Rm7zuU06+EKmiMy3XkJsc8yknV4gUqUxLMTQiu5JfQK1PWc4QKXKZFj98REbLaduKbIUwpvQnzfmcVS0izBlkBH6wrUhUqUV84DriE0NNGKfwGvAKSXOdtaxLQcJoQFA/6JtmxA8u4C1lpTiYRozYYVuRfIR1CoNOY990ISPwrW1FosoxMnvYWNTuOasMfiAPcFqKlNVG7mRARdbu5aoy+EZxD3DS6ajQ1+5VBSRnGHmwMw70mrYB4DxijDvADqAO2Gk+5/t+HNmRJICnyK7kMvAVyazsCUjv0JCvysDPEfqMck1AclLm7FZlsAa8R57HLgK/zZHv+03gInAfuGTkDJjzy4B0DhXpQiG3FPqDIuQ1I/4zNhnlVtxXzRmKrwJoQVbyWWKSUU6SeXg9g5TQVkwJRTmYRAzYW+iPymYakcqAP8Bey7qUlaCSCVeQtNMY4ugVj3xCpm+XbUWiyAkydX3bLetSdoKYwj3mPAT8DUBerKhBQpYNZP+qeOQCYrzPthWxhd8pnJ6+g34ViSN1SFltCtn7Kh5Jb90q4pWBcuFM26ffdExa1ShC5Erbz+G/JDcWZL8cuA/ZulV82WxQuKXtG8m8cxtLvIQxbi8HVjvalALEtvI9KGJb+R4ksa18VxRFURRFURRFCQ//AWdiZQc/sNBDAAAAAElFTkSuQmCC",
    # Cuboid: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAABmJLR0QA/wD/AP+gvaeTAAADl0lEQVRoge2Zz0tUURTHP0YmCFJQVLtIKaWwTdkuMigQKgwyatUysk2kZVChFpS1bFlkVAhWm35o+SfY71VgEggpwVRmamrmTNri3odnnu/O3Pvm6SjMFw7cOW/uOef77rvvnnMe5JBDDjlkiG1ABzCq5RlQntWIQmAnMA7M+GQMqMhiXM54hwo8DtzTktC611mMywnrmV2Ba0LfIvRro3a6LGJ7hcBJ8XvAMD6l/7voUAjUATGS90Q/cAA4iCIir8X0nEVBqAA4AXxl7sa2le/AeRaA0AodsEQhcIa5K+BJN3AFGBa6Ya3rNswxrVCBjiE0yoAXqDdPAvgAVFsQqBI2ioBdWoqEvsqC0CHtM6Fj6ARKXUlsAoYMjmwI2CIVoSD5CZS4OHisJ04Dd4AmYDBCAn6YCP0AGnUM01r30MXwLz2pXehqhIO3REPAjypt2/NzWFx7pHVDQRPTnSOjhvEA0OscZnr0knzejBjG1niJYj+J2nhHtBO57FNAK1AcxoEPG1GPz5TPxyfUk1AP/NW6ThfDO4AJ0m++TAmZCJhkAtju6qQS6BNGJlEbr9Xg2ERotRZbAp6dRu3T0/cBu11JeGgWhj4KfbEFoWN6jpx/1IKAvBFyfnNYEqmIeNgA3CT5zrnKFPAAdXb5YU0k0+z3C3Aa2ALcRZ3AEqNAAyqX+u27FtdzyoDjwOcMY0mJdCviRzHQJuacE9cahL4Nu5fDgq2IH33AWfE7z/C/ev3fyLA8SmMaMdSZUwpcEnpv3AN8i9rpfBABlSF3oDLeG0L/D7UakSPqR8tDF6oqjAldDFUtds2TTyP2MPdAvID5uQ+Cl0XP6LEt8oCLzD0QKx1sAKr39Ifg977LwRSWyGWD7wlU+mQNf9JYg0rgPGP+lMOEMETWMHsTe7TvOmZXxylp9OqRW0K3TwS119JOGCImP7fJoB5ZKcarxLiF+SusZFMvyP+Mi0GvGptGJXLNhCt1bVckVanbhEplvFK33WAjECWoQt828TMRSkfEtfkwiCoBnLAZdajFtbxHtYOCOoomQiYiNu2gau0zjsqQnxOcIVsjn+AGXTpCV0n+tDCudWEadPmZELDBkmqZ2iBVE3s/KjXpx24FFgUKSa5hasW1WqFvZpESkFjHbMAtQn9d6CP/0DNfeIMKOAHcR9Xk3qe3V1mMyxkVqA+f/o09Roi+VLZRDjxFtTpHgCfA1qxGlEMOOSx9/Adr/gmznITHOgAAAABJRU5ErkJggg==",
    GraphNodes: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAABmJLR0QA/wD/AP+gvaeTAAADJklEQVRoge2ZvW4TQRSFPxM7EhUVERQhtDQgpQq4BARJGVFbvAAUqZER5MdxIppQUSHSpAblAVKkoyJPkCg4KLYoiCIhK0JQzB08Hs+ud2f/jPCRRrPavTN77u69c+7Owhj/ByaBJnACtICGnPvnsA78ttpaoYw80UKRvwdU5biV5Q0vZTRvRfqSca6c0b0yRZPB0FovlJEnyvTnSQOYKJRRQmhHMkdWOZI7inSkDGwAbUZYayookjq0lhlctQrXGlO1O8AmveVW4w2DJLeB18AO8Bnoyvl5ctIaG64nuQXcBWpC9oyeID5y2JvNdOQ4Rz+cqh3UqobND5STNZTTbx32yzn6wbGDZBcVLjtCdttBctOaR+dRx7CZyp5+DysMkly1bMqop2uqup1HJg7EbjZtsmGYokfwFEUySLWjCuInsVv0JeVTyE1LfwDc8b2xhUPpb/pO4COIM9If+d7UAe3ITIhN6h9rS/SW3CCU6a+Am4S//UWx+xhik7qAbskkSyE2pqrrthFiPys2X0JsQj/WfEJLx/FhiM1T6ReAB9Y5GxXj2i3gFb23VwLmUJXCdeOcxt+FxCfZoziicea6qYUG8EyOK0AduAacA0+AG5b9vnH8IQKHPpiJFkW84nwhakGcB+47xh2h3kgVlRNfpa3ikeyuRFsJsS+jnrSZlEFao2PfrBTOhfwc/aGUGK5Ei1LgRRFE0+GgSmEoouZIyert4yR4KX1N+LxH5UkmWMPvqeX2zR4V71CEfhEt0eIKYi7QqvsTuB1xTFxBzBzTwHch8jzGuDaDS2o7dXYRMQHsCYld4iW3dsRcUk/TJhgVL4TACXA15thCt0xN9f4GXKCS+6HHXHEEMXW41Hs34ZyFLL8u9e4knDM3R1xlvJnU3TxIpI04NY+ZT20GBU+X43qeOv27KHrft0MG+76TMmGL4ertyqeGcb3uuG7WT65VrZB/jGY+LThI6fYYJYpB11Pb9/Wtf3SYXBlid8FwIb1szZkrXKFVH3J9WOgV8o/RFrygZG5Ls3cj7cWgQcHVcVKdSE1nxv8QRw1J43KfZKGRdPwYI4s/NGpiOOSeqxwAAAAASUVORK5CYII=",
    Cuboid3d: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA/0lEQVRIieXVQUoDQRAF0EdwmQO4iXcQouAhXHgBb5CcwW3ceQUXXsClq2QX8AR6BhG3AdGF3VCOPTOdMYLoh4Kh+VW/u6vnF/8FY8zwkGKW1r6NfVzgCW+NeMEVDoYUPsINNqHgCmcpVmF9k7jTvqIjnOIuJL/iFicF/iGuG5u4xzn2SgLrQHzGApPes35wFikn56/jrjOOK4rVolgr3nXN/U6V+5S/WwW6kmuaXCWQMeSZbiWQMcY8cObaf7RBAoM5ow7iTvA3Bboa2IX8EFqx9NkqLn21ilKTJ4kbrWJZEqgxuyiwtdlF9NnAILsu4ccGThN5ZD6m2NnI/P14BxLdhg05U8R5AAAAAElFTkSuQmCC",  # "zmdi zmdi-codepen"
    # ClosedSurfaceMesh: "data:image/png;base64,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",
    # ClosedSurfaceMesh: "data:image/png;base64,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",
    ClosedSurfaceMesh: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAACl0lEQVRYhe3WT4jVVRQH8M/4B50SQXIxg9iihsRFQjkuoihdVCQoUlCunKChZZQLSRRJamB03WYMAqVFi4JJaaOI2KJGotmILkboD/mHyqIZnJGaZp6Le368+9783rw/03K+cHn3nXPuOd/fveece1nGMprjZUyj0uaYw0dLDd6NHzsIXoxZbFsKgQ/C0e/x+y92N7B9CJ+rfv14zC91GvxRaevn8QKGwuEMnq2z3YzvQz+FfdiQEd/fCYGzsfhM/O/CJyG7i60hfx6/hXwik8NgyO9gfTvBd8fCSfRm8lX4KnS/4Jh0zpUgXB9kBa6E/kSrwdfgRiw6WKLfJO1AnmwnIlgZdkg58Q+2tELgaDi9htWZ/GmMKC/JaZzDATxc4vNU2F1oFrxIvAp2YW04HcuCzeHr7P+4WjJ/S3mzR9pN2Ig/Q//qYgS+CKPvMIw/Msd3cRKPhW0hhz4cbkLm3ZD9LJXsArxk4dZWpCQaiN3IkRPI0Yf38UOdn6ls/mEZgauZwQw+RX+ZYRMCOR5vQOa+qK48c5s56wR/SUc3KTW0Al1lxi/6f46gG29gVCq9wi6fDzVinCfhSbX1vlgSrsIrUsLlZ/0fzuNNHFJtYKVJSG0Z7oyvHlDtZmVl+LFqzy/GGN5BT/h9JPuY1xoFL3BEeSParnEjquC6dHs+UeJzRIuNiNQ8JmLBeyX6Xgu/+Pgi/vq12YpJ51lcRj2ZvAufhe4Wvoz5vNS4Vtb5yS+j4VaDFxiNhaczWfEmmMJTIXtbeqhUcLmOcHEd/4p17RKof5C8pZrZe+tsn8Pt0N/EM2ofJK+3G7zAsXDwk+q9P9jAthffqNb8tzG/2GlwahOykzGLJ5dCgNQh73UQfF4qy2UsY1E8AG0sM2O2giC0AAAAAElFTkSuQmCC",
    # Point3d: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAABqUlEQVRYhe3TsWtUQRDH8U/UHEQwFzQqGhGsLAKChY0pUtiLiiD+B6L+CSLYmNjam04btdLGTgStgo0iUUs5IwcWXrBR8Cx2l6yX9+4luTuC8L6w3DEzO/N7M7PU1NT8h5zBfbzDD/zEJzzA/CgLH8IjdCvOCxwfdvET+BwLfMMdnEYTe3EKt7AaY1ajbSCO4TE6+B0TP8X+Pnea8U4XX3B4kOLf/dvaJ9i9ibu7MhEPtysgJXiGGRzEROZv4B6+ooWFaEs00cYfnNyOgE4UMFPiX7Rx+e72xNy2vjPPcVXoTl8msJQlbZbEtaL/LObi/1ZPzGyByFeYLis+Jsy5K7zr830Ut2PcXCagXZBvXzwXsBLj3mC8KOmVGPARB8pURopGsFhxZyoTca0o4GV0Xq5IBHuExUvFF6KtikvWu7CB9Oyqvj4nCdgskzG+kwz5jNNcfm0h4VYZ6zXkAj7E33MjFJByvy9y3hTasyIszLCZEha8i+tFAQ28zURcFGY2KJPC8qXiy0qeIRzNRIziLONIleIGbghPZW0IRdfwWmh76ZfX1NTsGH8BaEidOSWP5WoAAAAASUVORK5CYII=",
}

type_to_shape_text = {
    AnyGeometry: "any shape",
    Rectangle: "rectangle",
    Polygon: "polygon",
    Bitmap: "bitmap (mask)",
    AlphaMask: "alpha mask",
    Polyline: "polyline",
    Point: "point",
    Cuboid: "cuboid",  #
    Cuboid3d: "cuboid 3d",
    Pointcloud: "pointcloud",  #  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "n-channel mask",  # "zmdi zmdi-collection-item"
    Point3d: "point 3d",  # "zmdi zmdi-select-all"
    GraphNodes: "keypoints",
    ClosedSurfaceMesh: "volume (3d mask)",
}


class ObjectClassView(Widget):
    def __init__(
        self,
        obj_class: ObjClass,
        show_shape_text: bool = True,
        show_shape_icon: bool = False,
        widget_id: str = None,
    ):
        self._obj_class = obj_class
        self._show_shape_text = show_shape_text
        self._show_shape_icon = show_shape_icon
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = self._obj_class.to_json()
        res["icon"] = None
        res["icon8"] = None
        if self._show_shape_icon is True:
            res["icon"] = type_to_zmdi_icon.get(self._obj_class.geometry_type)
            res["icon8"] = type_to_icons8_icon.get(self._obj_class.geometry_type)
        res["shape_text"] = None
        if self._show_shape_text is True:
            res["shape_text"] = type_to_shape_text.get(
                self._obj_class.geometry_type
            ).upper()
        return res

    def get_json_state(self):
        return None

"""
Asset viewer component for displaying S3 assets (images, etc.).

Provides:
- PNG/image display from S3 URLs
- Reactive updates based on selected record
- Loading states
"""

import logging
from typing import Callable, Optional

import panel as pn
import s3fs

logger = logging.getLogger(__name__)

# Public S3 filesystem for open-access buckets
FS_PUBLIC = s3fs.S3FileSystem(anon=True)


def get_s3_image_url(s3_path: str) -> Optional[str]:
    """
    Convert S3 path to HTTPS URL for public buckets.

    Args:
        s3_path: S3 path (s3://bucket/key or just bucket/key)

    Returns:
        HTTPS URL or None if invalid
    """
    if not s3_path:
        return None

    # Remove s3:// prefix if present
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]

    # Convert to HTTPS URL
    return f"https://{s3_path.split('/')[0]}.s3.amazonaws.com/{'/'.join(s3_path.split('/')[1:])}"


def check_s3_exists(s3_path: str) -> bool:
    """
    Check if an S3 path exists.

    Args:
        s3_path: S3 path to check

    Returns:
        True if exists, False otherwise
    """
    try:
        return FS_PUBLIC.exists(s3_path)
    except Exception as e:
        logger.warning(f"Error checking S3 path {s3_path}: {e}")
        return False


class AssetViewer:
    """
    Component for viewing assets from S3.

    Supports PNG images with automatic URL construction
    from S3 paths stored in DataFrame.
    """

    def __init__(
        self,
        s3_location_column: str = "S3_location",
        asset_filename: str = "fitted_session.png",
        width: int = 800,
        info_columns: Optional[list[str]] = None,
    ):
        """
        Initialize the asset viewer.

        Args:
            s3_location_column: Column containing S3 base path
            asset_filename: Filename of the asset within S3 location
            width: Display width for images
            info_columns: Columns to display in asset info panel (default: subject_id, session_date, agent_alias, n_trials)
        """
        self.s3_location_column = s3_location_column
        self.asset_filename = asset_filename
        self.width = width
        self.info_columns = info_columns or ["subject_id", "session_date", "agent_alias", "n_trials"]

    def get_asset_url(self, s3_location: str) -> Optional[str]:
        """
        Construct the full asset URL from S3 location.

        Args:
            s3_location: Base S3 path for the record

        Returns:
            Full HTTPS URL to the asset
        """
        if not s3_location:
            return None

        # Ensure path ends with /
        base = s3_location if s3_location.endswith("/") else f"{s3_location}/"
        full_path = f"{base}{self.asset_filename}"

        return get_s3_image_url(full_path)

    def create_viewer(
        self,
        record_ids_param,
        df_param,
        id_column: str = "_id",
        columns_param=None,
        highlights_param=None,
    ) -> pn.viewable.Viewable:
        """
        Create a reactive asset viewer for multiple records.

        Args:
            record_ids_param: Parameter containing list of selected record IDs
            df_param: Parameter containing the DataFrame
            id_column: Column name for record ID

        Returns:
            Panel Column that updates when selection changes
        """
        def render_assets(record_ids, df, columns=1, highlights=None):
            if not record_ids or df is None or df.empty:
                return pn.pane.Markdown(
                    "Select one or more records to view their assets.",
                    css_classes=["alert", "alert-info", "p-3"],
                )

            columns = max(1, int(columns or 1))
            use_divider = columns == 1

            # Render assets for all selected records
            asset_panels = []
            for record_id in record_ids:
                # Find the record
                mask = df[id_column].astype(str) == str(record_id)
                if not mask.any():
                    asset_panels.append(
                        pn.pane.Markdown(
                            f"Record {record_id} not found.",
                            css_classes=["alert", "alert-warning", "p-3"],
                        )
                    )
                    continue

                record = df[mask].iloc[0]
                s3_location = record.get(self.s3_location_column, "")

                if not s3_location:
                    asset_panels.append(
                        pn.pane.Markdown(
                            f"No S3 location available for record {record_id}.",
                            css_classes=["alert", "alert-warning", "p-3"],
                        )
                    )
                    continue

                asset_url = self.get_asset_url(s3_location)

                if not asset_url:
                    asset_panels.append(
                        pn.pane.Markdown(
                            f"Could not construct asset URL for record {record_id}.",
                            css_classes=["alert", "alert-danger", "p-3"],
                        )
                    )
                    continue

                # Build info panel
                info_items = []
                for col in self.info_columns:
                    if col in record.index:
                        info_items.append(f"**{col}:** {record[col]}")

                highlight_items = []
                if highlights:
                    for col in highlights:
                        if col in record.index:
                            value = record[col]
                            if isinstance(value, (int, float)):
                                value = f"{value:.3f}"
                            highlight_items.append(f"**{col}:** {value}")

                subtitle_parts = []
                if highlight_items:
                    subtitle_parts.append(
                        f"<span style=\"color: #c62828;\">{' | '.join(highlight_items)}</span>"
                    )
                if info_items:
                    subtitle_parts.append(" | ".join(info_items))
                subtitle = "\n\n".join(subtitle_parts) if subtitle_parts else None

                # Add this record's asset panel
                record_panel = pn.Column(
                    pn.pane.Markdown(f"### Record: {record_id}"),
                    pn.pane.Markdown(subtitle) if subtitle else None,
                    pn.pane.PNG(asset_url, width=self.width, alt_text=f"Asset for {record_id}"),
                    pn.layout.Divider() if use_divider else None,
                    sizing_mode="stretch_width",
                )
                asset_panels.append(record_panel)

            if not asset_panels:
                return pn.pane.Markdown(
                    "No assets to display.",
                    css_classes=["alert", "alert-warning", "p-3"],
                )

            if columns > 1:
                return pn.GridBox(*asset_panels, ncols=columns, sizing_mode="stretch_width")

            return pn.Column(*asset_panels, sizing_mode="stretch_width")

        if columns_param is None:
            columns_param = 1
        if highlights_param is None:
            highlights_param = []

        return pn.bind(
            render_assets,
            record_ids_param,
            df_param,
            columns=columns_param,
            highlights=highlights_param,
        )


def create_image_tooltip(
    s3_url_column: str = "asset_url",
    width: int = 600,
) -> str:
    """
    Create HTML tooltip template for Bokeh hover with S3 image.

    Args:
        s3_url_column: Column name containing the image URL
        width: Image width in pixels

    Returns:
        HTML template string for Bokeh HoverTool
    """
    return f"""
        <div style="text-align: left; white-space: nowrap;
                    border: 2px solid black; padding: 10px; background: white;">
            <img src="@{{{s3_url_column}}}{{safe}}"
                 style="width: {width}px; height: auto;"
                 alt="Asset preview">
        </div>
    """

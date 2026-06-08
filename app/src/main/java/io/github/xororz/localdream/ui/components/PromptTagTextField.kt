package io.github.xororz.localdream.ui.components

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Redo
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Backspace
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import androidx.compose.ui.window.PopupProperties
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.FuzzyMatcher
import io.github.xororz.localdream.data.TagMatchType
import io.github.xororz.localdream.data.TagSuggestion
import io.github.xororz.localdream.data.tagUnderscoresToSpaces

// Fixed height of the pinned action toolbar, subtracted from the suggestion
// list's height budget so the popup as a whole stays the same size as before.
private val SuggestionToolbarHeight = 52.dp

@Composable
fun PromptTagTextField(
    value: TextFieldValue,
    onValueChange: (TextFieldValue) -> Unit,
    label: @Composable (() -> Unit),
    suggestions: List<TagSuggestion>,
    onSuggestionClick: (TagSuggestion) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    showSuggestions: Boolean = true,
    onFocusChanged: (Boolean) -> Unit = {},
    onDismissSuggestions: () -> Unit = {},
    showToolbar: Boolean = false,
    onAddTag: () -> Unit = {},
    onClearTag: () -> Unit = {},
    onIncreaseWeight: () -> Unit = {},
    onDecreaseWeight: () -> Unit = {},
    onUndo: () -> Unit = {},
    onRedo: () -> Unit = {},
    undoEnabled: Boolean = false,
    redoEnabled: Boolean = false,
    highlightQuery: String? = null,
    overflowOffset: Int = -1,
    maxCollapsedLines: Int = 2,
    minCollapsedLines: Int = 2,
    minExpandedLines: Int = 3,
) {
    var expanded by remember { mutableStateOf(false) }
    var anchorWidthPx by remember { mutableIntStateOf(0) }
    var anchorTopPx by remember { mutableFloatStateOf(0f) }
    val density = LocalDensity.current

    // Grey out the portion of the prompt past the CLIP token limit. The offset
    // is a UTF-16 index from the backend; it may lag the latest keystroke, so
    // the transformation clamps it to the current text length.
    val overflowColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
    val overflowTransformation = remember(overflowOffset, overflowColor) {
        VisualTransformation { text ->
            if (overflowOffset in 0 until text.length) {
                val styled = buildAnnotatedString {
                    append(text.subSequence(0, overflowOffset))
                    withStyle(SpanStyle(color = overflowColor)) {
                        append(text.subSequence(overflowOffset, text.length))
                    }
                }
                TransformedText(styled, OffsetMapping.Identity)
            } else {
                TransformedText(text, OffsetMapping.Identity)
            }
        }
    }

    Column(modifier = modifier) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier
                .fillMaxWidth()
                .onFocusChanged { onFocusChanged(it.isFocused) }
                .onGloballyPositioned { coords ->
                    anchorWidthPx = coords.size.width
                    anchorTopPx = coords.positionInWindow().y
                },
            enabled = enabled,
            label = label,
            visualTransformation = overflowTransformation,
            maxLines = if (expanded) Int.MAX_VALUE else maxCollapsedLines,
            minLines = if (expanded) minExpandedLines else minCollapsedLines,
            shape = MaterialTheme.shapes.medium,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = MaterialTheme.colorScheme.primary,
                unfocusedBorderColor = MaterialTheme.colorScheme.outline,
            ),
            trailingIcon = {
                IconButton(onClick = { expanded = !expanded }) {
                    Icon(
                        imageVector = if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                        contentDescription = null,
                    )
                }
            },
        )
    }

    // The popup stays up for the toolbar even once the dictionary has no matches
    // (e.g. right after a completion is applied, or after a weight is wrapped),
    // so it does not blink away between edits.
    if (showSuggestions && (suggestions.isNotEmpty() || showToolbar) && anchorWidthPx > 0) {
        // Back gesture closes the suggestion popup. The Popup itself is not
        // focusable (so it never steals IME focus), so dismissOnBackPress never
        // fires; this BackHandler is what actually catches the gesture.
        BackHandler(enabled = true) { onDismissSuggestions() }

        // Reset the scroll to the top whenever the suggestion list changes. This
        // has to fire after the new list is applied: LazyColumn keyed items keep
        // the previously-anchored row pinned, so a freshly promoted top match
        // (e.g. "underwater" overtaking "underwear") would otherwise stay scrolled
        // off the top of the viewport.
        val listState = rememberLazyListState()
        LaunchedEffect(suggestions) {
            listState.scrollToItem(0)
        }

        val widthDp = with(density) { anchorWidthPx.toDp() }
        val gapDp = 8.dp
        val gapPx = with(density) { gapDp.toPx() }

        // Insets observed live so the popup tracks IME open/close animations.
        val imeBottomPx = WindowInsets.ime.getBottom(density)
        val statusTopPx = WindowInsets.statusBars.getTop(density)
        val navBottomPx = WindowInsets.navigationBars.getBottom(density)
        val bottomInsetPx = maxOf(imeBottomPx, navBottomPx)

        // Space above the field that is actually visible (excludes the status bar).
        val availableAbovePx = (anchorTopPx - statusTopPx - gapPx).coerceAtLeast(0f)
        // The toolbar and its divider are fixed chrome on top of the list, so the
        // list cap is the original budget minus that chrome. This keeps the total
        // popup height unchanged from before the toolbar existed, so it still
        // never covers the prompt text above the field.
        val capDp = with(density) { minOf(280.dp, availableAbovePx.toDp()) }
        val maxHeightDp = (capDp - SuggestionToolbarHeight - 1.dp).coerceAtLeast(0.dp)
        Popup(
            popupPositionProvider = remember(statusTopPx, bottomInsetPx) {
                AnchorPositionProvider(
                    safeTopPx = statusTopPx,
                    bottomInsetPx = bottomInsetPx,
                )
            },
            properties = PopupProperties(
                focusable = false,
                dismissOnBackPress = true,
                dismissOnClickOutside = false,
            ),
        ) {
            Card(
                modifier = Modifier.width(widthDp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
            ) {
                // Fixed action row pinned above the scrolling list so the actions
                // stay reachable no matter how far the user scrolls.
                TagActionToolbar(
                    onAddTag = onAddTag,
                    onClearTag = onClearTag,
                    onIncreaseWeight = onIncreaseWeight,
                    onDecreaseWeight = onDecreaseWeight,
                    onUndo = onUndo,
                    onRedo = onRedo,
                    undoEnabled = undoEnabled,
                    redoEnabled = redoEnabled,
                )
                if (suggestions.isNotEmpty()) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = maxHeightDp),
                    ) {
                        items(
                            items = suggestions,
                            key = { it.replacementTag },
                        ) { suggestion ->
                            SuggestionRow(
                                suggestion = suggestion,
                                highlightQuery = highlightQuery,
                                onClick = { onSuggestionClick(suggestion) },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TagActionToolbar(
    onAddTag: () -> Unit,
    onClearTag: () -> Unit,
    onIncreaseWeight: () -> Unit,
    onDecreaseWeight: () -> Unit,
    onUndo: () -> Unit,
    onRedo: () -> Unit,
    undoEnabled: Boolean,
    redoEnabled: Boolean,
) {
    // Three tonal groups read apart at a glance: tag edits (secondary), weight
    // steps (tertiary) and the undo/redo history (primary). This mirrors the MD3
    // Expressive button-group look without depending on experimental APIs.
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(SuggestionToolbarHeight)
            .padding(horizontal = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ToolbarAction(
            icon = Icons.Default.Add,
            contentDescription = "add tag",
            container = MaterialTheme.colorScheme.secondaryContainer,
            onColor = MaterialTheme.colorScheme.onSecondaryContainer,
            onClick = onAddTag,
        )
        ToolbarAction(
            icon = Icons.Default.Backspace,
            contentDescription = "clear tag",
            container = MaterialTheme.colorScheme.secondaryContainer,
            onColor = MaterialTheme.colorScheme.onSecondaryContainer,
            onClick = onClearTag,
        )
        ToolbarAction(
            icon = Icons.Default.ArrowUpward,
            contentDescription = "increase weight",
            container = MaterialTheme.colorScheme.tertiaryContainer,
            onColor = MaterialTheme.colorScheme.onTertiaryContainer,
            onClick = onIncreaseWeight,
        )
        ToolbarAction(
            icon = Icons.Default.ArrowDownward,
            contentDescription = "decrease weight",
            container = MaterialTheme.colorScheme.tertiaryContainer,
            onColor = MaterialTheme.colorScheme.onTertiaryContainer,
            onClick = onDecreaseWeight,
        )
        ToolbarAction(
            icon = Icons.AutoMirrored.Filled.Undo,
            contentDescription = "undo",
            container = MaterialTheme.colorScheme.primaryContainer,
            onColor = MaterialTheme.colorScheme.onPrimaryContainer,
            onClick = onUndo,
            enabled = undoEnabled,
        )
        ToolbarAction(
            icon = Icons.AutoMirrored.Filled.Redo,
            contentDescription = "redo",
            container = MaterialTheme.colorScheme.primaryContainer,
            onColor = MaterialTheme.colorScheme.onPrimaryContainer,
            onClick = onRedo,
            enabled = redoEnabled,
        )
    }
}

@Composable
private fun RowScope.ToolbarAction(
    icon: ImageVector,
    contentDescription: String,
    container: Color,
    onColor: Color,
    onClick: () -> Unit,
    enabled: Boolean = true,
) {
    val disabledAlpha = 0.38f
    Surface(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .weight(1f)
            .fillMaxHeight()
            .padding(vertical = 6.dp),
        shape = MaterialTheme.shapes.large,
        color = if (enabled) container else container.copy(alpha = disabledAlpha),
        contentColor = if (enabled) onColor else onColor.copy(alpha = disabledAlpha),
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(
                imageVector = icon,
                contentDescription = contentDescription,
                modifier = Modifier.size(20.dp),
            )
        }
    }
}

@Composable
private fun SuggestionRow(suggestion: TagSuggestion, highlightQuery: String?, onClick: () -> Unit) {
    // Embedding names round-trip into the prompt verbatim; spaces would break
    // the lookup in PromptProcessor (it keys on the lowercase filename stem).
    val displayPrimary = if (suggestion.matchType == TagMatchType.Embedding) {
        suggestion.primaryText
    } else {
        tagUnderscoresToSpaces(suggestion.primaryText)
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(
                    color = if (suggestion.matchType == TagMatchType.Embedding) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        categoryColor(suggestion.category)
                    },
                    shape = CircleShape,
                ),
        )
        Spacer(Modifier.width(10.dp))
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                text = highlightMatches(displayPrimary, highlightQuery, MaterialTheme.colorScheme.primary),
                style = MaterialTheme.typography.titleSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            suggestion.secondaryText?.takeIf { it.isNotBlank() }?.let { secondary ->
                Text(
                    text = highlightMatches(secondary, highlightQuery, MaterialTheme.colorScheme.primary),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        if (suggestion.postCount > 0) {
            Spacer(Modifier.width(8.dp))
            Text(
                text = formatPostCount(suggestion.postCount),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        MatchTypeBadge(suggestion.matchType)
    }
}

@Composable
private fun MatchTypeBadge(matchType: TagMatchType) {
    val label = when (matchType) {
        TagMatchType.Alias -> stringResource(R.string.tag_alias_label)
        TagMatchType.Correction -> stringResource(R.string.tag_correction_label)
        TagMatchType.Embedding -> stringResource(R.string.tag_embedding_label)
        else -> return
    }
    val container = when (matchType) {
        TagMatchType.Correction -> MaterialTheme.colorScheme.errorContainer
        TagMatchType.Embedding -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.secondaryContainer
    }
    val onContainer = when (matchType) {
        TagMatchType.Correction -> MaterialTheme.colorScheme.onErrorContainer
        TagMatchType.Embedding -> MaterialTheme.colorScheme.onPrimaryContainer
        else -> MaterialTheme.colorScheme.onSecondaryContainer
    }
    Spacer(Modifier.width(8.dp))
    Card(
        shape = MaterialTheme.shapes.small,
        colors = CardDefaults.cardColors(containerColor = container),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = onContainer,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 1.dp),
        )
    }
}

@Composable
private fun categoryColor(category: Int): Color = when (category) {
    1 -> Color(0xFFE53935)

    // artist
    3 -> Color(0xFFAB47BC)

    // copyright
    4 -> Color(0xFF43A047)

    // character
    5 -> Color(0xFFFB8C00)

    // meta
    else -> MaterialTheme.colorScheme.outline // general / unknown
}

private fun formatPostCount(n: Int): String = when {
    n >= 1_000_000 -> "%.1fM".format(n / 1_000_000.0)
    n >= 10_000 -> "${n / 1_000}k"
    n >= 1_000 -> "%.1fk".format(n / 1_000.0)
    else -> n.toString()
}

private fun normalizeForHighlight(value: String): String = value.lowercase().replace(' ', '_').replace('-', '_')

private fun highlightMatches(text: String, query: String?, highlightColor: Color): AnnotatedString {
    if (query.isNullOrBlank()) return AnnotatedString(text)
    val normQuery = normalizeForHighlight(query.trim())
    if (normQuery.isEmpty()) return AnnotatedString(text)
    // normalizeForHighlight only swaps single chars (no trimming or collapsing),
    // so positions into normText line up one-to-one with text.
    val normText = normalizeForHighlight(text)
    val positions = FuzzyMatcher.positions(normQuery.toCharArray(), normText)
    if (positions == null || positions.isEmpty()) return AnnotatedString(text)
    val matchStyle = SpanStyle(fontWeight = FontWeight.Bold, color = highlightColor)
    return buildAnnotatedString {
        var idx = 0
        var p = 0
        while (idx < text.length) {
            if (p < positions.size && positions[p] == idx) {
                val start = idx
                while (p < positions.size && positions[p] == idx) {
                    p++
                    idx++
                }
                withStyle(matchStyle) {
                    append(text.substring(start, idx))
                }
            } else {
                val start = idx
                while (idx < text.length && (p >= positions.size || positions[p] != idx)) idx++
                append(text.substring(start, idx))
            }
        }
    }
}

private class AnchorPositionProvider(private val safeTopPx: Int, private val bottomInsetPx: Int) : PopupPositionProvider {
    override fun calculatePosition(
        anchorBounds: IntRect,
        windowSize: IntSize,
        layoutDirection: LayoutDirection,
        popupContentSize: IntSize,
    ): IntOffset {
        val gap = 8
        // Visible bottom is the lowest screen Y that is not occluded by the IME
        // or the navigation bar. Falling back to windowSize.height keeps the math
        // safe if insets are reported as 0 on some OEMs.
        val visibleBottom = (windowSize.height - bottomInsetPx).coerceAtLeast(0)

        val aboveY = anchorBounds.top - popupContentSize.height - gap
        val belowY = anchorBounds.bottom + gap

        val y = when {
            // Prefer above when it sits below the safe top — this is the normal
            // path; the popup never overlaps the caret because the height was
            // already capped to the available space above.
            aboveY >= safeTopPx -> aboveY

            // Otherwise drop below only when it fully fits inside the visible
            // region (i.e. above the keyboard / nav bar). Without this guard
            // the popup can render behind the IME on edge-to-edge windows
            // where windowSize.height is the full screen.
            belowY + popupContentSize.height <= visibleBottom -> belowY

            // Last resort: clamp above the field as high as the safe top allows.
            // heightIn already prevents the popup from being taller than the
            // space above, so this still avoids covering the caret.
            else -> aboveY.coerceAtLeast(safeTopPx)
        }
        val x = anchorBounds.left.coerceIn(
            0,
            (windowSize.width - popupContentSize.width).coerceAtLeast(0),
        )
        return IntOffset(x, y)
    }
}
